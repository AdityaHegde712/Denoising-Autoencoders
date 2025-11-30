"""
Visualizes a few samples from the Natural Image Dataset
Displays
  - Noisy image
  - Clean ground truth
  - Noise (difference)
    
"""

import torch
import matplotlib.pyplot as plt
from utils import get_data

def show_samples(train_path, validate_path, test_path, batch_size=4, num_samples=3):

    # Load dataloader from utils.py
    train_loader, _, _ = get_data(
        train_path=train_path,
        validate_path=validate_path,
        test_path=test_path,
        batch_size=batch_size,
    )

    noisy_batch, clean_batch, noise_batch = next(iter(train_loader))

    # Convert tensor to numpy for plotting
    noisy_batch = noisy_batch.permute(0, 2, 3, 1).cpu().numpy()
    clean_batch = clean_batch.permute(0, 2, 3, 1).cpu().numpy()
    noise_batch = noise_batch.permute(0, 2, 3, 1).cpu().numpy()

    num_samples = min(num_samples, noisy_batch.shape[0])

    plt.figure(figsize=(12, num_samples * 4))

    for i in range(num_samples):
        # noisy image
        plt.subplot(num_samples, 3, 3*i + 1)
        plt.imshow(noisy_batch[i])
        plt.title("Noisy Image")
        plt.axis("off")

        # clean image
        plt.subplot(num_samples, 3, 3*i + 2)
        plt.imshow(clean_batch[i])
        plt.title("Clean (Ground Truth)")
        plt.axis("off")

        # noise difference (normalize to be between 0 to 1)
        noise_vis = (noise_batch[i] - noise_batch[i].min()) / (noise_batch[i].max() - noise_batch[i].min())

        plt.subplot(num_samples, 3, 3*i + 3)
        plt.imshow(noise_vis)
        plt.title("Noise (Difference)")
        plt.axis("off")

    plt.show()
