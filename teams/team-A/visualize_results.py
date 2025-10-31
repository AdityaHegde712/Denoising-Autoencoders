"""
Result visualisation script using the log dictionary produced during training.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from train import plot_training_history


def import_results(log_path: str):
    """
    Imports training log from JSON file.
    
    Args:
        log_path: Path to the JSON log file.
    Returns:
        log_dict: Dictionary with training history.
    """
    with open(log_path, 'r') as f:
        log_dict = json.load(f)
    
    return log_dict


def main():
    log_dict = import_results('./runs/20251019_045235/training_log.json')
    plot_training_history(log_dict, save_path="./runs/20251019_045235/training_curves.png")


if __name__ == "__main__":
    main()
