import cv2
import torch 
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

def evaluate_ensemble(pred_dir, label_dir, dest_dir): 

    # Obtain the pred and labels 
    pred_paths = sorted([pred_dir+"/"+i for i in os.listdir(pred_dir+"/")])
    label_paths = sorted([label_dir+"/"+i for i in os.listdir(label_dir+"/")])

    total_dice = 0             
    for i, pred_path in enumerate(tqdm(pred_paths)):
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) 
        label = cv2.imread(label_paths[i], cv2.IMREAD_GRAYSCALE) 

        total_dice+=dice_metric(pred, label)
    
    dice_score = total_dice/len(pred_paths)

    print(f"\nThe dice score is {dice_score}")

if __name__ == "__main__": 
    PRED_PATH = ""
    LABEL_PATH = ""
    
    evaluate_ensemble(PRED_PATH, LABEL_PATH)
