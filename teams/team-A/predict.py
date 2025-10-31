"""
Script to run inference with the ConvolutionalAutoencoder model.
"""

import os
import sys
import argparse
from glob import glob
from typing import List, Union

import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm.auto import tqdm

# ---- adjust these imports to your project structure ----
# e.g., if these live in src/models.py, use: from src.models import ...
from model import Encoder, Decoder, Autoencoder, ConvolutionalAutoencoder, TrainConfig

# -----------------------------
# Utilities
# -----------------------------
def load_image(path: str, size=(512, 512)) -> torch.Tensor:
    """
    Loads an RGB image, resizes to size, converts to [0,1] tensor [3,H,W].
    """
    tfm = T.Compose([
        T.Resize(size),
        T.ToTensor(),         # => [0,1]
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img) # type: ignore

def save_side_by_side(inp: torch.Tensor, out: torch.Tensor, save_path: str, title: str = "") -> None:
    """
    inp/out: [3,H,W] tensors in [0,1]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    inp_np = inp.permute(1, 2, 0).cpu().numpy()
    out_np = out.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 4))
    if title:
        plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.imshow(inp_np)
    plt.axis("off")
    plt.title("Input (noisy)")

    plt.subplot(1, 2, 2)
    plt.imshow(out_np)
    plt.axis("off")
    plt.title("Output (denoised)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def ensure_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return list(x)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference with ConvolutionalAutoencoder.")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint .pt", default="./runs/fully_conv_1/best.pt")
    parser.add_argument("--imagedir", type=str, required=False, help="Path to images directory", default="./predict_images/")
    parser.add_argument("--outdir", type=str, required=False, help="Where to save side-by-side PNGs", default="./predict_images/outputs/")
    args = parser.parse_args()

    image_paths = ensure_list(list(glob(os.path.join(args.imagedir, "*.*"))))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build a model instance and load weights
    # Note: ConvolutionalAutoencoder __init__ expects loaders; pass None (not used in predict)
    cfg = TrainConfig(device=device)
    empty_loader = DataLoader(torch.utils.data.TensorDataset(torch.zeros(1,3,512,512)))
    cae = ConvolutionalAutoencoder(
        train_loader=empty_loader,
        val_loader=empty_loader,
        test_loader=empty_loader,
        cfg=cfg,
    )
    # Load checkpoint (relies on your class' load_checkpoint implementation)
    cae.load_checkpoint(
        path=args.ckpt,
        model=cae.model,
        scheduler=None,
        scaler=None,
        map_location=device,
    )
    cae.model.eval()

    # Inference over images
    os.makedirs(args.outdir, exist_ok=True)
    for img_path in tqdm(image_paths, desc="Inference"):
        if not os.path.isfile(img_path):
            print(f"[warn] Skipping missing file: {img_path}")
            continue

        x = load_image(img_path)     # [3,H,W], [0,1]
        y = cae.predict(x)                                  # [1,3,H,W] on CPU
        y = y.squeeze(0).clamp(0, 1)                        # [3,H,W]

        base = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(args.outdir, f"{base}_side_by_side.png")
        save_side_by_side(x, y, save_path, title=base)
        # print(f"[saved] {save_path}")

    print("Done.")

if __name__ == "__main__":
    # Allow running without CLI by editing variables above and calling: python inference.py --ckpt path/to/best.pt
    main()
