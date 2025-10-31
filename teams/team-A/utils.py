'''
Utility functions for Team A runs
'''

import os
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import torchvision.transforms as transforms

from PIL import Image


def seed_all(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remove_noise_prefix(fn):
    """
    Helper function for prefix removal
    """
    prefixes = ["gauss_", "poisson_", "salt and pepper_", "speckle_"]

    for prefix in prefixes:
        if fn.startswith(prefix):
            return fn.split("_", 1)[1]

    return fn


def remove_nids_prefix(dir):
    """
    Removes all prefixes for the NISN Dataset.
    """
    new_fn = ""
    for fn in tqdm(os.listdir(dir)):
        old_fn_dir = os.path.join(dir, fn)
        new_fn = remove_noise_prefix(fn)
        new_fn_dir = os.path.join(dir, new_fn)

        if old_fn_dir != new_fn_dir:
            os.rename(old_fn_dir, new_fn_dir)
        else:
            print(f'Skipping file: {old_fn_dir} as it either the prefix has already been removed or it does not have a prefix.')


class NaturalImageDataSet(Dataset):
    """
    Creates Dataset class for Natural Image Data Set.
    """

    def __init__(self, dir, transform=None):
        self.noisy_dir = os.path.join(dir, "noisy images")
        self.clean_dir = os.path.join(dir, "ground truth")
        self.transform = transform

        self.images = os.listdir(self.noisy_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imagefn = self.images[index]
        noisy_image_fp = os.path.join(self.noisy_dir, imagefn)
        clean_image_fp = os.path.join(self.clean_dir, imagefn)

        # print(f"Paths:\n{noisy_image_fp}\n{clean_image_fp}")

        # Make sure file exists. Otherwise, return an error.
        try:
            noisy_image = Image.open(noisy_image_fp).convert("RGB")
            clean_image = Image.open(clean_image_fp).convert("RGB")
        except FileNotFoundError:
            print(f'File Not Found: Unable to load matching ground truth image for the following image: {noisy_image_fp}')
            print(f'Missing ground truth image: {clean_image_fp}')
            return None, None

        if self.transform:  # Apply transforms here.
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image


def get_data(
    train_path: str,
    validate_path: str,
    test_path: str,
    batch_size: int = 8,
):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5)
    ])


    training_data = NaturalImageDataSet(train_path, transform=transform)
    training_NIDS_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    #May or may not need to be included, since the base CAE already has dataloaders.

    test_data = NaturalImageDataSet(test_path, transform=transform)
    test_NIDS_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    validation_data = NaturalImageDataSet(validate_path, transform=transform)
    validate_NIDS_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return training_NIDS_loader, test_NIDS_loader, validate_NIDS_loader


class AverageMeter:
    """Tracks running average of a scalar."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)
