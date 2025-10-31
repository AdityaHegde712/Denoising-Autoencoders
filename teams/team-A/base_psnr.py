'''
This script computes the PSNR between noisy and clean images in the training dataset.
'''

#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from tqdm import tqdm

# import your project util
from utils import get_data  # get_data(train_path, validate_path, test_path)

TRAIN_PATH = './../../data/nisn/train/train'
TEST_PATH = './../../data/nisn/test/test'
VALIDATE_PATH = './../../data/nisn/validate/validate'

def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    """PSNR in dB assuming targets in [0,1]"""
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))

def main():
    # Build loaders (we only use train here)
    _, _, loader = get_data(TRAIN_PATH, VALIDATE_PATH, TEST_PATH, batch_size=8)

    psnrs = []
    with torch.no_grad():
        pbar = tqdm(loader, dynamic_ncols=True, desc="PSNR(noisy vs clean)")
        for noisy, clean in pbar:
            # MSE per sample
            bsz = clean.size(0)
            mse = F.mse_loss(noisy, clean, reduction="none").view(bsz, -1).mean(dim=1)
            psnr = psnr_from_mse(mse)  # [B]
            psnrs.append(psnr.cpu())
            pbar.set_postfix(batch_psnr=f"{psnr.mean().item():.2f} dB")

    if psnrs:
        psnrs = torch.cat(psnrs, dim=0)
        print(f"\nTrain PSNR (noisy→clean): mean={psnrs.mean().item():.2f} dB, "
              f"std={psnrs.std(unbiased=False).item():.2f} dB, n={psnrs.numel()}")
    else:
        print("No samples found in train loader.")

if __name__ == "__main__":
    main()
