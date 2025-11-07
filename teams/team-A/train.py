'''
Main training script for Team A's model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as Datasets
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
from tqdm import tqdm as tqdm_regular
import seaborn as sns
from torchvision.utils import make_grid
import random
import os
from PIL import Image
import pandas as pd
from model import ConvolutionalAutoencoder, TrainConfig
from utils import get_data
from datetime import datetime
import json

TRAIN_PATH = './../../data/nisn/train/train'
TEST_PATH = './../../data/nisn/test/test'
VALIDATE_PATH = './../../data/nisn/validate/validate'


def plot_training_history(log_dict, save_path="./outputs/training_curves.png"):
    """
    Create graphs showing how the model improved during training.
    
    This creates two graphs:
    1. Loss over time (training and validation)
    2. PSNR over time (training and validation)
    3. SSIM over time if applicable (training and validation)
    
    Args:
        log_dict: Dictionary with training history
        save_path: Where to save the graph image
    """
    # Convert dictionary to pandas DataFrame for easier plotting
    df = pd.DataFrame.from_dict(log_dict, orient='index')
    df.index = df.index.astype(int)  # Make sure epoch numbers are integers
    df = df.sort_index()  # Sort by epoch number
    
    # Check if validation metrics exist
    has_val_metrics = len([col for col in df.columns if col.startswith('val_')]) >= 2
    
    # Create a figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Loss over epochs
    axes[0].plot(df.index, df['train_loss'], label='Training Loss', marker='o', linewidth=2)
    if has_val_metrics:
        axes[0].plot(df.index, df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss over Training', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: PSNR over epochs
    axes[1].plot(df.index, df['train_psnr'], label='Training PSNR', marker='o', linewidth=2)
    if has_val_metrics:
        axes[1].plot(df.index, df['val_psnr'], label='Validation PSNR', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('PSNR over Training', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: SSIM over epochs if available
    if 'train_ssim' in df.columns:
        axes[2].plot(df.index, df['train_ssim'], label='Training SSIM', marker='o', linewidth=2)
        if has_val_metrics and 'val_ssim' in df.columns:
            axes[2].plot(df.index, df['val_ssim'], label='Validation SSIM', marker='s', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('SSIM', fontsize=12)
        axes[2].set_title('SSIM over Training', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis('off')  # Hide the third plot if SSIM is not available
        
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    
    # Display the plot in the notebook
    plt.show()
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Best Training Loss: {df['loss'].min():.6f} (Epoch {df['train_loss'].idxmin()})")
    if has_val_metrics:
        print(f"Best Validation Loss: {df['val_loss'].min():.6f} (Epoch {df['val_loss'].idxmin()})")
    print(f"Best Training PSNR: {df['psnr'].max():.2f} dB (Epoch {df['psnr'].idxmax()})")
    if has_val_metrics:
        print(f"Best Validation PSNR: {df['val_psnr'].max():.2f} dB (Epoch {df['val_psnr'].idxmax()})")


def main():
    print(os.getcwd())
    # Device assignment
    device = None
    if torch.cuda.is_available():
        device = "cuda"
        print('Running on the GPU')
    else:
        device = "cpu"
        print('Running on the CPU')
    
    run_name = input("Enter name for the training run: ")
    run_name = run_name if run_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Setting up training run:", run_name)
    print()

    cfg = TrainConfig(
        epochs=100,
        batch_size=8,
        lr=3e-4,
        weight_decay=0,
        optimizer='adam',

        scheduler='cosine',
        warmup_epochs=5,

        accum_steps=1,
        early_stopping=True,
        early_stopping_patience=15,
    
        loss='l1',
        seed=42,
        device=device,
        out_dir=f"./runs/{run_name}/",
        resume=False
    )

    # Set up training configs/dataloaders
    train_loader, test_loader, val_loader = get_data(TRAIN_PATH, VALIDATE_PATH, TEST_PATH, batch_size=cfg.batch_size)

    # Model, optimizer, loss
    model = ConvolutionalAutoencoder(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # Train
    try:
        stats, log_dict = model.train()
    except Exception as e:
        print("An error occurred during training:", str(e))
        print("Stacktrace: ", str(e.__traceback__))
        if os.listdir(cfg.out_dir):
            print(f"Partial outputs saved to: {cfg.out_dir}")
        else:
            print("No outputs were saved.")
            os.rmdir(cfg.out_dir)  # Remove empty output directory
        return
    print("Final stats:", stats)

    # Save log dict as JSON
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(log_dict, f, indent=4)
    print(f"Training log saved to: {log_path}")

    # # Plot training history
    plot_training_history(log_dict, save_path=os.path.join(cfg.out_dir, "training_curves.png"))


if __name__ == "__main__":
    main()
