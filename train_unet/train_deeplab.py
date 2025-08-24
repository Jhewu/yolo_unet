import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from tiny_unet.tiny_unet import TinyUNet
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from dataset import CustomDataset
from torch.amp import GradScaler

from torchinfo import summary
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

# class TverskyFocalCombinedLoss(torch.nn.Module):
#     def __init__(self, alpha=0.7, beta=0.3, focal_gamma=2.0, tversky_weight=0.5, focal_weight=0.5, smooth=1e-8, ema_decay=0.9):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.focal_gamma = focal_gamma
#         self.tversky_weight = tversky_weight
#         self.focal_weight = focal_weight
#         self.smooth = smooth
#         self.ema_decay = ema_decay

#         self.register_buffer('tversky_ema', torch.tensor(0.0))
#         self.register_buffer('focal_ema', torch.tensor(0.0))
#         self.register_buffer('initialized', torch.tensor(False))

#     def forward(self, inputs, targets):
#         inputs = torch.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         # Tversky Loss
#         TP = (inputs * targets).sum()
#         FP = ((1 - targets) * inputs).sum()
#         FN = (targets * (1 - inputs)).sum()
#         tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
#         tversky_loss = 1 - tversky

#         # Focal Loss
#         focal_loss = torchvision.ops.sigmoid_focal_loss(inputs, targets, gamma=self.focal_gamma, reduction='mean')

#         # Update EMA
#         if self.training:
#             if not self.initialized:
#                 self.tversky_ema = tversky_loss.detach().clone()
#                 self.focal_ema = focal_loss.detach().clone()
#                 self.initialized.fill_(True)
#             else:
#                 self.tversky_ema.mul_(self.ema_decay).add_(tversky_loss.detach(), alpha=1 - self.ema_decay)
#                 self.focal_ema.mul_(self.ema_decay).add_(focal_loss.detach(), alpha=1 - self.ema_decay)

#         # Normalize using EMA
#         ema_min = 1e-6
#         tversky_norm = tversky_loss / (max(self.tversky_ema, ema_min) + 1e-8)
#         focal_norm = focal_loss / (max(self.focal_ema, ema_min) + 1e-8)

#         # Combined loss
#         total_weight = self.tversky_weight + self.focal_weight
#         loss = (self.tversky_weight * tversky_norm + self.focal_weight * focal_norm) / total_weight

#         return loss

# class TverskyFocalCombinedLoss(torch.nn.Module):
#     def __init__(self, alpha=0.7, beta=0.3, focal_gamma=2.0, tversky_weight=0.7, focal_weight=0.3, smooth=1e-8):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta

#         self.focal_gamma = focal_gamma

#         self.tversky_weight = tversky_weight
#         self.focal_weight = focal_weight

#         self.smooth = smooth

#     def forward(self, inputs, targets):
#         # Apply sigmoid
#         inputs = torch.sigmoid(inputs)
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         # Tversky Loss
#         TP = (inputs * targets).sum()
#         FP = ((1 - targets) * inputs).sum()
#         FN = (targets * (1 - inputs)).sum()
#         tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
#         tversky_loss = 1 - tversky

#         # Focal Loss (from PyTorch)
#         focal_loss = torchvision.ops.sigmoid_focal_loss(inputs, targets, gamma=self.focal_gamma, reduction='mean')

#         # print(f"\nTversky Loss: {tversky_loss.item()}")
#         # print(f"Focal Loss: {focal_loss.item()}\n")

#         # Combined
#         loss = self.tversky_weight * tversky_loss + self.focal_weight * focal_loss
#         return loss

# class TverskyLoss(torch.nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(TverskyLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1, alpha=0.7, beta=0.3):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = torch.nn.functional.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         #True Positives, False Positives & False Negatives
#         TP = (inputs * targets).sum()    
#         FP = ((1-targets) * inputs).sum()
#         FN = (targets * (1-inputs)).sum()
       
#         Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
#         return 1 - Tversky

class DiceFocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, gamma=1.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # === Dice Loss ===
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        # === Focal Tversky Loss ===
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky_loss = (1 - tversky) ** self.gamma

        # === Combined ===
        loss = 0.5 * dice_loss + 0.5 * focal_tversky_loss
        return loss

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=2):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.nn.functional.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name) 

def GetCurrentTime(): 
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

def plot_loss_curves(history, save_path): 
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", color="blue")
    plt.plot(history["val_loss"], label="Validation Loss", color="red")
    plt.plot(history["train_dice_metric"], label="Training DICE Score", color="orange")
    plt.plot(history["val_dice_metric"], label="Validation DICE Score", color="green")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Metric")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "plot.png"))
    plt.show()

def dice_metric(pred, target, smooth=1e-8):
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
    pred = torch.nn.functional.sigmoid(pred)
    
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

def dice_loss(pred, target, smooth=1e-8):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W) - logits
        target: Tensor of ground truth (batch_size, 1, H, W) - binary masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Loss (lower is better)
    """
    dice_score = dice_metric(pred, target, smooth)
    return 1 - dice_score

def train_unet(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = CustomDataset(DATA_PATH, "images/train", "labels/train", IMAGE_SIZE)
    val_dataset = CustomDataset(DATA_PATH, "images/test", "labels/test", IMAGE_SIZE)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # Grab the original conv1
    orig_conv1 = model.backbone.conv1

    # Build a new conv that takes 4 channels
    new_conv1 = torch.nn.Conv2d(
        in_channels=4,                    # <-- changed
        out_channels=orig_conv1.out_channels,
        kernel_size=orig_conv1.kernel_size,
        stride=orig_conv1.stride,
        padding=orig_conv1.padding,
        bias=orig_conv1.bias is not None   # preserve bias flag
    )

    # --- initialize the new conv ---------------------------------
    with torch.no_grad():
        # Copy the pre‑trained weights for the first 3 channels
        new_conv1.weight[:, :3] = orig_conv1.weight

        # Decide how to initialise the 4th channel
        # 1) Zero init (most common)
        new_conv1.weight[:, 3:4] = torch.zeros_like(orig_conv1.weight[:, :1])

    # Swap the conv in the model
    model.backbone.conv1 = new_conv1

    # Replace the final 1x1 conv to output 1 channel
    new_classifier = torch.nn.Sequential(
        # keep everything before the final conv
        *list(model.classifier.children())[:-1],          # all layers except the last conv
        torch.nn.Conv2d(
            in_channels=256,
            out_channels=1,        # <‑‑ change here
            kernel_size=1,
            stride=1,
            padding=0
        )
    )

    # Assign the new head back to the model
    model.classifier = new_classifier

    if LOAD_AND_TRAIN: 
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

    summary(model, input_size=(BATCH_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE))

    # Initialize the optimizer, to adjust the parameters of a model and minimize the loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LOAD_AND_TRAIN:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=int(PATIENCE * 0.5), verbose=True)
    else: scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # For Mixed-Precision Training
    scaler = GradScaler("cuda")

    # Initialize variables for callbacks
    history = dict(train_loss=[], val_loss=[], train_dice_metric=[], val_dice_metric=[])
    best_val_loss = float("inf")

    dest_dir = f"runs/unet_{GetCurrentTime()}" 
    model_dir = os.path.join(dest_dir, "weights")
    CreateDir(model_dir)

    # Initialize local patience variable for early stopping
    patience = 0

    combined_loss = DiceFocalTverskyLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()

        start_time = time.time()
        train_running_loss = 0
        train_running_dice_metric = 0

        if MIX_PRECISION:
            for idx, img_mask in enumerate(tqdm(train_dataloader)):
                with torch.amp.autocast(device_type="cuda"): 
                    img = img_mask[0].float().to(device)
                    mask = img_mask[1].float().to(device)

                    y_pred = model(img)["out"]
                    loss = combined_loss(y_pred, mask)
                    metric = dice_metric(y_pred, mask)

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                train_running_loss += loss.item()
                train_running_dice_metric += metric.item()

        else:
            for idx, img_mask in enumerate(tqdm(train_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)["out"]
                optimizer.zero_grad()

                loss = dice_loss(y_pred, mask)
                metric = dice_metric(y_pred, mask)

                train_running_loss += loss.item()
                train_running_dice_metric += metric.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        end_time = time.time()
        train_loss = train_running_loss / (idx + 1)
        train_dice_metric = train_running_dice_metric / (idx + 1)

        model.eval()
        val_running_loss = 0
        val_running_dice_metric = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)["out"]
                loss = dice_loss(y_pred, mask)
                val_metric = dice_metric(y_pred, mask)

                val_running_loss += loss.item()
                val_running_dice_metric += val_metric.item()

            val_loss = val_running_loss / (idx + 1)
            val_dice_metric = val_running_dice_metric / (idx + 1)
        
        # Update the scheduler
        if LOAD_AND_TRAIN:
            scheduler.step(val_loss)
        else: scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice_metric"].append(val_dice_metric)
        history["train_dice_metric"].append(train_dice_metric)

        if val_loss < best_val_loss: 
            if (best_val_loss - val_loss) > 1e-3:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(os.path.join(model_dir, "best.pth")))
                patience = 0
            else: 
                print(f"Validation loss improved slightly from {best_val_loss:.4f} to {val_loss:.4f}, but not significantly enough to save the model.")
                if epoch+1 >= EARLY_STOPPING_START: 
                    patience+=1
        else:
            if epoch+1 >= EARLY_STOPPING_START: 
                patience+=1
        
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(dest_dir, "history.csv"), index=False)

        print("-"*30)
        print(f"This is Patience {patience}")
        print(f"Training Speed per EPOCH (in seconds): {end_time - start_time:.4f}")
        print(f"Maximum Gigabytes of VRAM Used: {torch.cuda.max_memory_reserved(device) * 1e-9:.4f}")
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print(f"Train DICE Score EPOCH {epoch+1}: {train_dice_metric:.4f}")
        print(f"Valid DICE Score EPOCH {epoch+1}: {val_dice_metric:.4f}")
        print("-"*30)

        if patience >= PATIENCE: 
            print(f"\nEARLY STOPPING: Valid Loss did not improve since epoch {epoch+1-patience}, terminating training...")
            break

    torch.save(model.state_dict(), os.path.join(os.path.join(model_dir, "last.pth")))
    plot_loss_curves(history, dest_dir)

if __name__ == "__main__":
    IMAGE_SIZE = 128
    MIX_PRECISION = True
    DATA_PATH = "yolo_cropped_verified"
    WIDTHS = [32, 64, 128, 256]
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    EPOCHS = 50

    LOAD_AND_TRAIN = False
    MODEL_PATH = "runs/2025_08_11_00_53_48_yolo_cropped/weights/best.pth"

    EARLY_STOPPING_START = 20
    PATIENCE = 5

    train_unet()
    