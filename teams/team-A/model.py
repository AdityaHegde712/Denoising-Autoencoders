"""
Script to define the functions and classes relevant to the main autoencoder for Team A.
"""

import os
from tqdm.auto import tqdm
import math
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from torchvision.utils import make_grid
import random
import os
import json
from PIL import Image

from utils import *

# Variables
LATENT_DIMS = 512  # Size of latent representation


@dataclass
class TrainConfig:
    # Core
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # one of {"adam", "adamw", "sgd"}

    # Scheduler
    scheduler: str = "none"  # or "cosine"
    warmup_epochs: int = 0

    # Optimization niceties
    # amp: bool = True  # automatic mixed precision
    # grad_clip_norm: float = 1.0  # 0 or None to disable
    accum_steps: int = 1  # gradient accumulation steps
    early_stopping: bool = False
    early_stopping_patience: int = 10

    # Repro / IO
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "./outputs"
    ckpt_last: str = "last.pt"
    ckpt_best: str = "best.pt"
    resume: Optional[bool] = False

    # Loss
    loss: str = "mse"  # {"mse", "l1", "charbonnier"}
    charbonnier_eps: float = 1e-3


def create_optimizer(model: nn.Module, cfg: TrainConfig) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9, nesterov=True)
    raise ValueError("Unsupported optimizer: %s" % cfg.optimizer)


def create_scheduler(opt: Optimizer, cfg: TrainConfig, steps_per_epoch: int):
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        total_epochs = cfg.epochs
        if cfg.warmup_epochs > 0:
            warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_epochs * steps_per_epoch)
            cosine = CosineAnnealingLR(opt, T_max=max(1, (total_epochs - cfg.warmup_epochs) * steps_per_epoch))
            return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs * steps_per_epoch])
        else:
            return CosineAnnealingLR(opt, T_max=max(1, total_epochs * steps_per_epoch))
    raise ValueError("Unsupported scheduler: %s" % cfg.scheduler)


def create_loss(cfg: TrainConfig):
    if cfg.loss == "mse":
        return nn.MSELoss(reduction='mean')
    if cfg.loss == "l1":
        return nn.L1Loss(reduction='mean')
    if cfg.loss == "charbonnier":
        class Charbonnier(nn.Module):
            def __init__(self, eps: float = 1e-3):
                super().__init__()
                self.eps = eps

            def forward(self, pred, target):
                diff = pred - target
                return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return Charbonnier(cfg.charbonnier_eps)
    raise ValueError("Unsupported loss: %s" % cfg.loss)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    """PSNR in dB assuming targets in [0,1]"""
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


def enblock(in_channels: int, out_channels: int, stride: int = 2,
            act_fn: nn.Module = nn.SiLU(inplace=True)) -> nn.Sequential:
    # Use 8 groups when divisible; otherwise fall back to InstanceNorm-like GroupNorm(1, C)
    g = 8 if out_channels % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                  padding=1, padding_mode='reflect', bias=False),
        nn.GroupNorm(g, out_channels),
        act_fn,
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                  padding=1, padding_mode='reflect', bias=False),
        nn.GroupNorm(g, out_channels),
        act_fn,
    )


def deblock(in_channels: int, out_channels: int, stride: int = 2,
            act_fn: nn.Module = nn.SiLU(inplace=True)) -> nn.Sequential:
    # First ConvTranspose2d upsamples; prefer k=4, s=2, p=1 (output_padding=0) to avoid checkerboard
    k1, s1, p1, op1 = (4, 2, 1, 0) if stride == 2 else (3, 1, 1, 0)
    g = 8 if out_channels % 8 == 0 else 1
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k1, stride=s1,
                           padding=p1, output_padding=op1, bias=False),
        nn.GroupNorm(g, out_channels),
        act_fn,
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1,
                           padding=1, output_padding=0, bias=False),
        nn.GroupNorm(g, out_channels),
        act_fn,
    )


# Define Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32, latent_dim: int = LATENT_DIMS, act_fn: nn.Module = nn.SiLU(inplace=True)):
        super().__init__()
        stride = 2
        assert isinstance(act_fn, nn.Module), "act_fn must be a nn.Module"
    
        C1 = out_channels
        C2, C3, C4, C5 = C1*2, C1*4, C1*8, C1*16

        # 6× downsample: 512→256→128→64→32→16→8
        self.net = nn.Sequential(
            # If my math is right
            enblock(in_channels, C1, stride),  # 512→256
            enblock(C1, C2, stride),  # 256→128
            enblock(C2, C3, stride),  # 128→64
            enblock(C3, C4, stride),  # 64→32
            enblock(C4, C5, stride),  # 32→16
            enblock(C5, C5, stride),  # 16→8
        )
        
        self.latent_channels = latent_dim
        self.to_latent = nn.Conv2d(C5, self.latent_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.net(x)
        z = self.to_latent(x)
        return z


# Define Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 3, latent_dim: int = LATENT_DIMS):
        super().__init__()
        stride = 2

        C1 = in_channels
        C2, C3, C4, C5 = C1*2, C1*4, C1*8, C1*16

        self.latent_channels = latent_dim
        self.from_latent = nn.Conv2d(self.latent_channels, C5, kernel_size=1, bias=False)

        self.net = nn.Sequential(
            deblock(C5, C5, stride),  # 8→16
            deblock(C5, C4, stride),  # 16→32
            deblock(C4, C3, stride),  # 32→64
            deblock(C3, C2, stride),  # 64→128
            deblock(C2, C1, stride),  # 128→256
        )
        
        self.head = nn.Sequential(
            nn.ConvTranspose2d(C1, out_channels, kernel_size=4, stride=2, padding=1, bias=False),  # 256→512
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.from_latent(z)
        x = self.net(x)
        x = self.head(x)
        return x


# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


#  defining wrapper class
class ConvolutionalAutoencoder():
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: TrainConfig = TrainConfig(),
    ):
        self.model = Autoencoder(Encoder(), Decoder()).to(cfg.device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        self.model.train()

        # Training logs
        train_loss_meter = AverageMeter()
        train_psnr_meter = AverageMeter()
        
        # Scheduler step per epoch
        scheduler = create_scheduler(optimizer, self.cfg, steps_per_epoch=len(train_loader))

        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            dynamic_ncols=True,
            leave=True,
            mininterval=0.2
        )
        for step, (noisy, clean) in pbar:
            noisy = noisy.to(self.cfg.device, non_blocking=True)
            clean = clean.to(self.cfg.device, non_blocking=True)

            # Forward pass
            recon = self.model(noisy)

            # Loss calculation
            loss = loss_fn(recon, clean)

            # Gradient calculation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Optimizer step
            optimizer.step()
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Metrics
            with torch.no_grad():
                bs = clean.size(0)

                batch_mse = F.mse_loss(recon, clean, reduction='none').view(bs, -1).mean(dim=1)
                batch_psnr = psnr_from_mse(batch_mse).mean().item()
                
                train_loss_meter.update(loss.item() * max(1, self.cfg.accum_steps), n=bs)
                train_psnr_meter.update(batch_psnr, n=bs)
            
            # Update pbar
            pbar.set_description(
            f"train {step}/{len(train_loader)}"
            )
            pbar.set_postfix(
                loss=f"{train_loss_meter.avg:.4f}",
                psnr=f"{train_psnr_meter.avg:.2f} dB"
            )

        return {
            "train_loss": train_loss_meter.avg,
            "train_psnr": train_psnr_meter.avg,
        }


    def train(
        self,
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        seed_all(self.cfg.seed)

        optimizer = create_optimizer(self.model, self.cfg)
        loss_fn = create_loss(self.cfg)

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        last_path = os.path.join(self.cfg.out_dir, self.cfg.ckpt_last)
        best_path = os.path.join(self.cfg.out_dir, self.cfg.ckpt_best)

        # Save cfg
        cfg_dict = asdict(self.cfg)
        # JSON
        with open(os.path.join(self.cfg.out_dir, "train_config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2, sort_keys=True)

        start_epoch = 1
        best_val_loss = float("inf")

        # Tracking setup
        log_dict = {}
        no_improve_epochs = 0

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            # Train
            train_stats = self._train_one_epoch(self.train_loader, optimizer, loss_fn)

            # Validate
            val_stats = self.eval(loss_fn)

            # Logging
            log_dict[epoch] = {**train_stats, **val_stats}

            # Checkpoints (last + best), and early stopping
            self.save_checkpoint(last_path, optimizer, epoch, best_val_loss)
            if val_stats["val_loss"] < best_val_loss:
                best_val_loss = val_stats["val_loss"]
                self.save_checkpoint(best_path, optimizer, epoch, best_val_loss)
                print(f"[Best] Val loss improved to {best_val_loss:.6f} -> saved {best_path}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if self.cfg.early_stopping and no_improve_epochs >= self.cfg.early_stopping_patience:
                    print(f"[Early stopping] No improvement for {no_improve_epochs} epochs "
                        f"(patience={self.cfg.early_stopping_patience}). Stopping training.")
                    break

            # Pbar separator
            print()

        print("Training complete.")
        return {"best_val_loss": best_val_loss}, log_dict

    def eval(self, loss_fn) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.val_loader, start=1),
                total=len(self.val_loader),
                dynamic_ncols=True,
                leave=True,
                mininterval=0.2
            )
            for step, (noisy, clean) in pbar:
                noisy = noisy.to(self.cfg.device, non_blocking=True)
                clean = clean.to(self.cfg.device, non_blocking=True)
                
                recon = self.model(noisy)
                loss = loss_fn(recon, clean)
                
                batch_mse = F.mse_loss(recon, clean, reduction='none').view(clean.size(0), -1).mean(dim=1)
                batch_psnr = psnr_from_mse(batch_mse).mean().item()
                
                loss_meter.update(loss.item(), n=clean.size(0))
                psnr_meter.update(batch_psnr, n=clean.size(0))
                
                # Update pbar
                pbar.set_description(
                    f"eval  {step}/{len(self.test_loader)}"
                )
                pbar.set_postfix(
                    val_loss=f"{loss_meter.avg:.4f}",
                    val_psnr=f"{psnr_meter.avg:.2f} dB"
                )

        return {
            "val_loss": loss_meter.avg,
            "val_psnr": psnr_meter.avg
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a single image [3, H, W] or batch [B, 3, H, W].
        Assumes inputs are in [0,1]. Returns reconstructed tensor in [0,1] on CPU.
        """
        self.model.eval()
        with torch.inference_mode():
            # Ensure batch dimension
            if x.dim() == 3:
                x = x.unsqueeze(0)  # [1, 3, H, W]
            x = x.to(self.cfg.device, non_blocking=True)

            y = self.model(x)          # [B, 3, H, W]
            y = y.clamp(0.0, 1.0)      # ensure valid image range
            return y.detach().cpu()

    def save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_loss: float
    ) -> None:
        if not path:
            path = os.path.join(self.cfg.out_dir, self.cfg.ckpt_last)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            # "scheduler": None if scheduler is None else scheduler.state_dict(),
            # "scaler": None if scaler is None else scaler.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": asdict(self.cfg),
            "best_val": best_val_loss,
        }, path)


    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        scheduler: Any,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        map_location: Optional[str] = None
    ) -> Tuple[int, float, TrainConfig]:
        ckpt = torch.load(path, map_location=map_location)
        model.load_state_dict(ckpt["model"])
        # if optimizer is not None and ckpt.get("optimizer") is not None:
        #     optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        cfg_dict = ckpt.get("cfg", {})
        cfg = TrainConfig(**cfg_dict) if cfg_dict else TrainConfig()
        return start_epoch, best_val, cfg


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate your modules (using your exact class signatures/defs)
    enc = Encoder(in_channels=3, out_channels=32, latent_dim=1500).to(device)
    dec = Decoder(in_channels=32, out_channels=3, latent_dim=1500).to(device)
    ae = Autoencoder(enc, dec).to(device)

    # eval-mode + no grad for a shape/range check
    ae.eval()
    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        # x = torch.rand(2, 3, 512, 512, device=device)
        imgs = [
            Image.fromarray(
                np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8),
                mode="RGB"
            )
            for _ in range(4)
        ]
        
        sample = np.array(imgs[0])  # For range checking later
        x = torch.stack([transform(img) for img in imgs], dim=0).to(device)  # type: ignore # (4,3,512,512) in [0,1]

        y = ae(x)
        print("input shape :", x.shape)
        print("output shape:", y.shape)
        print("original range: ", float(sample.min()), "→", float(sample.max()))
        print("output range:", float(y.min()), "→", float(y.max()))
        print("input range: ", float(x.min()), "→", float(x.max()))
        assert y.shape == (4, 3, 512, 512)
