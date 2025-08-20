import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from tiny_unet.tiny_unet import TinyUNet

from dataset import CustomDataset
from torch.amp import GradScaler

from torchinfo import summary
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name) 

def GetCurrentTime(): 
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

def plot_loss_curves(history, save_path): 
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss (DICE Loss)", color="blue")
    plt.plot(history["val_loss"], label="Validation Loss (DICE Loss)", color="red")
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

    model = TinyUNet(in_channels=4, num_classes=1).to(device)

    if LOAD_AND_TRAIN: 
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

    summary(model, input_size=(BATCH_SIZE, 4, IMAGE_SIZE, IMAGE_SIZE))

    # Initialize the optimizer, to adjust the parameters of a model and minimize the loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LOAD_AND_TRAIN:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=int(PATIENCE * 0.5), verbose=True
    )

    # Initiliaze the scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(EPOCHS * 0.25))

    # For Mixed-Precision Training
    scaler = GradScaler("cuda")

    # Initialize variables for callbacks
    history = dict(train_loss=[], val_loss=[], val_dice_metric=[])
    best_val_loss = float("inf")

    dest_dir = f"runs/unet_{GetCurrentTime()}" 
    model_dir = os.path.join(dest_dir, "weights")
    CreateDir(model_dir)

    # Initialize local patience variable for early stopping
    patience = 0

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
                    y_pred = model(img)
                    loss = dice_loss(y_pred, mask)
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

                y_pred = model(img)
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

        model.eval()
        val_running_loss = 0
        val_running_dice_metric = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
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

        if val_loss < best_val_loss: 
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(os.path.join(model_dir, "best.pth")))
            patience = 0
        else:
            if epoch+1 >= EARLY_STOPPING_START: 
                patience+=1
        
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(dest_dir, "history.csv"), index=False)

        print("-"*30)
        print(f"Training Speed per EPOCH (in seconds): {end_time - start_time:.4f}")
        print(f"Maximum Gigabytes of VRAM Used: {torch.cuda.max_memory_reserved(device) * 1e-9:.4f}")
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
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
    DATA_PATH = "yolo_cropped"
    WIDTHS = [32, 64, 128, 256]
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 100

    LOAD_AND_TRAIN = False
    MODEL_PATH = "runs/2025_08_11_00_53_48_yolo_cropped/weights/best.pth"

    EARLY_STOPPING_START = 30
    PATIENCE = 30

    train_unet()