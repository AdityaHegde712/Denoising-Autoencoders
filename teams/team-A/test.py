import json
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvolutionalAutoencoder, TrainConfig

TRAIN_PATH = './../../data/nisn/train/train'
TEST_PATH = './../../data/nisn/test/test'
VALIDATE_PATH = './../../data/nisn/validate/validate'

def visualize_results(example_noisy, example_denoised, example_clean, save_path="./outputs/test_results.png"):
    """
    Create a visual comparison of noisy, denoised, and clean images.
    
    This creates a grid showing:
    - Row 1: Noisy input images
    - Row 2: Model's denoised output
    - Row 3: Ground truth (actual clean images)
    
    Args:
        example_noisy: Tensor of noisy images
        example_denoised: Tensor of denoised images from model
        example_clean: Tensor of ground truth clean images
        save_path: Where to save the comparison image
    """
    
    def denormalize(img):
        """Convert images from [-1, 1] back to [0, 1] for display"""
        return (img + 1) / 2
    
    num_examples = example_noisy.size(0)
    
    # Create figure with 3 rows (noisy, denoised, clean) and multiple columns
    fig, axes = plt.subplots(3, num_examples, figsize=(num_examples * 2, 6))
    
    # If we only have one example, axes won't be 2D, so fix that
    if num_examples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_examples):
        # Denormalize images (convert from [-1,1] to [0,1])
        noisy_img = denormalize(example_noisy[i]).permute(1, 2, 0).numpy()
        denoised_img = denormalize(example_denoised[i]).permute(1, 2, 0).numpy()
        clean_img = denormalize(example_clean[i]).permute(1, 2, 0).numpy()
        
        # Clip to valid range [0, 1]
        noisy_img = np.clip(noisy_img, 0, 1)
        denoised_img = np.clip(denoised_img, 0, 1)
        clean_img = np.clip(clean_img, 0, 1)
        
        # Row 1: Noisy images
        axes[0, i].imshow(noisy_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Noisy Input', fontsize=12, fontweight='bold')
        
        # Row 2: Denoised images
        axes[1, i].imshow(denoised_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Model Output\n(Denoised)', fontsize=12, fontweight='bold')
        
        # Row 3: Clean ground truth
        axes[2, i].imshow(clean_img)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Ground Truth\n(Clean)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTest result visualization saved to: {save_path}")
    
    # Display in notebook
    plt.show()

def main():
    # Test the model on the test dataset
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)

    device = None
    if torch.cuda.is_available():
        device = "cuda"
        print('Running on the GPU')
    elif torch.backends.mps.is_available():
        device = "mps"
        print('Running on Metal')        
    else:
        device = "cpu"
        print('Running on the CPU')

    #Specify the run_name directory, so that we know where to get the best model.
    #If we can't find the directory, quit the test script.
    run_name = input("Enter run name directory to load in the best model: ")
    
    try:
        run_name = run_name
        print(f"Locating {run_name} directory.")

        cfg = TrainConfig(
            device=device,
            out_dir=f"./runs/{run_name}"        
        ) 

        if not os.path.isdir(cfg.out_dir):
            raise FileNotFoundError(f"Unable to locate {cfg.out_dir} directory.")

    except FileNotFoundError as e:
        print(e)
        return

    # We only need the test loader to get metrics and reports.
    _, _, test_loader = get_data(TRAIN_PATH, VALIDATE_PATH, TEST_PATH, batch_size=cfg.batch_size)

    model = ConvolutionalAutoencoder(
        cfg=cfg,
        train_loader=_,
        val_loader=_,
        test_loader=test_loader
    )    

    checkpoint_path = os.path.join(cfg.out_dir, "best.pt")

    output_path = os.path.join(cfg.out_dir, "outputs")

    os.makedirs(output_path, exist_ok=True) 
    try:
        test_results = model.test(checkpoint_path=checkpoint_path) # This is where fit() saved the best model
    except Exception as e:
        print("An error occurred during testing:", str(e))
        print("Stacktrace: ", str(e.__traceback__))
        if os.listdir(output_path):
            print(f"Partial outputs saved to: {output_path}")
        else:
            print("No outputs were saved.")
            os.rmdir(output_path)  # Remove empty output directory
        return    

    # Save test metrics. 
    metrics = {
        'test_loss': test_results['test_loss'],
        'test_psnr': test_results['test_psnr'],
        'test_ssim': test_results['test_ssim'],        
    }

    metrics_path = os.path.join(output_path, "test_metrics.json")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Test metrics saved to: {metrics_path}")        

    # Visualize the results
    print("\nCreating result visualizations...")
    visualize_results(
        test_results['example_noisy'],
        test_results['example_denoised'],
        test_results['example_clean'],
        save_path=os.path.join(output_path, "test_results.png")
    ) 

    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print(f"\nCheck the {cfg.out_dir} directory for:") 
    print("  - test_metrics.json (final test performance)")
    print("  - test_results.png (visual comparison of images)")


if __name__ == "__main__":
    main()    