import os
import cv2
import torch 
import numpy as np
from tqdm import tqdm

def dice_metric(pred, target, smooth=1):
    """
    Computes the Dice Score/Coefficient for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W) - logits or probabilities
        target: Tensor of ground truth (batch_size, 1, H, W) - binary masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Score (higher is better)
    """
    # Apply sigmoid to convert logits to probabilities if needed
    pred = torch.sigmoid(pred)
    
    # Flatten tensors for easier computation (optional but common)
    pred_flat = pred.view(pred.size(0), -1)  # (batch_size, H*W)
    target_flat = target.view(target.size(0), -1)  # (batch_size, H*W)
    
    # Calculate intersection
    intersection = (pred_flat * target_flat).sum(dim=1)  # Per batch
    
    # Calculate Dice coefficient
    # Formula: 2 * |A ∩ B| / (|A| + |B|)
    dice_denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + smooth) / (dice_denominator + smooth)
    
    return dice.mean()  # Average across batch

def evaluate_ensemble(pred_dir, label_dir): 

    # Obtain the pred and labels 
    # pred_paths = sorted([pred_dir+"/"+i for i in os.listdir(pred_dir+"/")])
    label_paths = sorted([label_dir+"/"+i for i in os.listdir(label_dir+"/")])

    total_dice = 0             
    for i, label_path in enumerate(tqdm(label_paths)):
        label_name = os.path.basename(label_path)
        pred_path = os.path.join(pred_dir, label_name)

        if os.path.exists(pred_path):
            pred = torch.from_numpy(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0)
        else: 
            pred = torch.zeros(IMAGE_SIZE, IMAGE_SIZE).unsqueeze(0)

        label = torch.from_numpy(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0)

        # Calculate dice score and add to total
        total_dice+=dice_metric(pred, label)
    
    dice_score = total_dice/len(label_path)

    print(f"\nThe dice score is {dice_score}")

if __name__ == "__main__": 
    PRED_PATH = "reconstructed_val/labels"
    LABEL_PATH = "stacked_segmentation/labels"
    SPLIT = "test"
    IMAGE_SIZE = 192
    
    evaluate_ensemble(os.path.join(PRED_PATH, SPLIT), os.path.join(LABEL_PATH, SPLIT))
