import os
import cv2
import torch 
import numpy as np
from tqdm import tqdm


def dice_metric(pred, target, smooth=1e-6):
    """
    Computes the Dice Score/Coefficient for binary segmentation.
    This function expects binary tensors (0 or 1).
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W) - binary masks
        target: Tensor of ground truth (batch_size, 1, H, W) - binary masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Score (higher is better)
    """
    # Flatten tensors for easier computation
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    # Calculate intersection
    # The multiplication works correctly on binary inputs
    intersection = (pred_flat * target_flat).sum(dim=1)
    
    # Calculate Dice coefficient
    dice_denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + smooth) / (dice_denominator + smooth)
    
    return dice.mean()

def evaluate_ensemble(pred_dir, label_dir):
    # Get label paths
    label_paths = sorted([os.path.join(label_dir, i) for i in os.listdir(label_dir)])

    total_dice = 0
    num_samples = len(label_paths)
    
    for i, label_path in enumerate(tqdm(label_paths)):
        label_name = os.path.basename(label_path)
        pred_path = os.path.join(pred_dir, label_name)

        pred = None
        # Load prediction
        if os.path.exists(pred_path):
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert to tensor and NORMALIZE to [0, 1] range
            pred = torch.from_numpy(pred).float() / 255.0
            
            # Add channel and batch dimension
            pred = pred.unsqueeze(0).unsqueeze(0)
            
        else:
            # Create empty mask if prediction doesn't exist
            # Assuming OG_IMG_SIZE is the size of your reconstructed masks
            pred = torch.zeros(1, 1, 192, 192) # Use correct size
            
        # Load label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to tensor and NORMALIZE to [0, 1] range
        label = torch.from_numpy(label).float() / 255.0
        
        # Add channel and batch dimension
        label = label.unsqueeze(0).unsqueeze(0)
        
        # Ensure binary masks for both prediction and label
        # A threshold of 0.5 works on normalized values
        pred_binary = (pred > 0.5).float()
        label_binary = (label > 0.5).float()

        # Calculate dice score on binary tensors
        dice = dice_metric(pred_binary, label_binary)
        print(f"Sample {i}: Dice = {dice.item()}")
        total_dice += dice.item()
        
    dice_score = total_dice / num_samples
    print(f"\nThe average dice score is {dice_score}")

if __name__ == "__main__":
    SPLIT = "test"
    PRED_PATH = f"reconstructed_{SPLIT}/labels"
    LABEL_PATH = "stacked_segmentation/labels"
    
    evaluate_ensemble(os.path.join(PRED_PATH, SPLIT), os.path.join(LABEL_PATH, SPLIT))

