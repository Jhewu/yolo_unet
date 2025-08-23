import torch 
import torch.nn as nn 
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.efficientnet import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torch.utils.data import WeightedRandomSampler

from torchvision import transforms, datasets
from torch.amp import GradScaler
from tqdm import tqdm

import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import torchvision

from verifier_net.verifier_net import VerificationNet

def GetCurrentTime(): 
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name) 

def plot_loss_curves(history, save_path): 
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", color="blue")
    plt.plot(history["train_unnorm_loss"], label="Training Unnormalized Loss", color="purple")
    plt.plot(history["val_loss"], label="Validation Loss", color="red")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Metric")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "plot.png"))
    plt.show()

if __name__ == "__main__": 
    EPOCH = 100
    BATCH = 256
    IMG_SIZE = 224
    EARLY_STOPPING_START = 20
    PATIENCE = 10
    LR = 1e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VerificationNet()
    model.train()
    model.to(device)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only train the classifier
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Create the train_dataset
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.Lambda(lambda x: x.convert('RGBA') if x.mode != 'RGBA' else x),
        transforms.ToTensor(), 
    ])

    train_dataset = datasets.ImageFolder(
        root="verifier_dataset/train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root="verifier_dataset/test",
        transform=transform
    )

    weights = [1/1200 if label == 0 else 1/500 for _, label in train_dataset]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        # sampler=sampler, 
        batch_size=BATCH, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=BATCH, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    scaler = GradScaler("cuda")
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.3).to(device))
    criterion_val = nn.BCEWithLogitsLoss()# pos_weight=torch.tensor(2.2).to(device))
    best_val_loss = float("inf")

    dest_dir = f"runs/unet_{GetCurrentTime()}" 
    model_dir = os.path.join(dest_dir, "weights")

    CreateDir(model_dir)

    history = dict(train_loss=[], val_loss=[], train_unnorm_loss=[])

    for epoch in tqdm(range(EPOCH)):

        start_time = time.time()
        train_running_loss = 0
        train_running_unnorm_loss = 0

        for idx, batch in enumerate(tqdm(train_dataloader)): 
            # Batch is (images, labels) from ImageFolder
            images, labels = batch  # <-- You were missing this!
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)  # Shape: (B, 1)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"): 
                # Forward pass
                y_pred = model(images)
                loss = criterion_train(y_pred, labels)
                unnorm_loss = criterion_val(y_pred, labels)
                # loss = torchvision.ops.sigmoid_focal_loss(y_pred, labels, gamma=2, alpha=1, reduction="mean")

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

                train_running_unnorm_loss += unnorm_loss.item()
                train_running_loss += loss.item()

        end_time = time.time()
        train_loss = train_running_loss / (idx + 1)
        train_unnorm_loss = train_running_unnorm_loss / (idx + 1)

        model.eval()
        val_running_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dataloader)):
                images, labels = batch  # <-- You were missing this!
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)  # Shape: (B, 1)

                y_pred = model(images)
                loss = criterion_val(y_pred, labels)
                # loss = torchvision.ops.sigmoid_focal_loss(y_pred, labels, reduction="mean")

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_unnorm_loss"].append(train_unnorm_loss)

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
        print(f"Training Speed per EPOCH (in seconds): {end_time - start_time:.4f}")
        print(f"Maximum Gigabytes of VRAM Used: {torch.cuda.max_memory_reserved(device) * 1e-9:.4f}")
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Train Unnormalized Loss EPOCH {epoch+1}: {train_unnorm_loss:.4f}")
        print(f"Val Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

        if patience >= PATIENCE: 
            print(f"\nEARLY STOPPING: Valid Loss did not improve since epoch {epoch+1-patience}, terminating training...")
            break

    torch.save(model.state_dict(), os.path.join(os.path.join(model_dir, "last.pth")))
    plot_loss_curves(history, dest_dir)




    