import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from dataset import CustomDataset
from torch.amp import GradScaler

from torchinfo import summary

def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()

def calculate_pos_weight(train_dataloader, device): 
    total_positive_pixels = 0
    total_negative_pixels = 0

    print("\nCalculating class weights...")

    for idx, img_mask in enumerate(tqdm(train_dataloader)):
        mask = img_mask[1].float()
        total_positive_pixels += torch.sum(mask == 1).item()
        total_negative_pixels += torch.sum(mask == 0).item()
    
    if total_positive_pixels == 0: 
        print("\n## Warning: No positive pixels found in the training data. pos_weight will be set to 1 ")
        pos_weight = torch.tensor(1.0).to(device)
    else: 
        pos_weight = torch.tensor(total_negative_pixels / total_positive_pixels).to(device)

    print(f"\nCalculated pos_weight: {pos_weight.item()}")
    return pos_weight


def train_unet(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CustomDataset(DATA_PATH, "images/train", "labels/train", 192)
    val_dataset = CustomDataset(DATA_PATH, "images/val", "labels/val", 192)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=4, widths=WIDTHS, num_classes=1).to(device)

    summary(model, input_size=(BATCH_SIZE, 4, 192, 192))

    # pos_weight = calculate_pos_weight(train_dataloader, device)
    pos_weight = torch.tensor(32.0).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler("cuda")

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0

        if MIX_PRECISION:
            for idx, img_mask in enumerate(tqdm(train_dataloader)):
                with torch.amp.autocast(device_type="cuda"): 
                    img = img_mask[0].float().to(device)
                    mask = img_mask[1].float().to(device)
                    y_pred = model(img)
                    # loss = criterion(y_pred, mask)
                    loss = dice_loss(y_pred, mask)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_running_loss += loss.item()

        else:
            for idx, img_mask in enumerate(tqdm(train_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                optimizer.zero_grad()

                # loss = criterion(y_pred, mask)
                loss = dice_loss(y_pred, mask)
                train_running_loss += loss.item()
                
                loss.backward()
                optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        model.eval()
        val_running_loss = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                # loss = criterion(y_pred, mask)
                loss = dice_loss(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 128
    WIDTHS = [32, 64, 128, 256]
    EPOCHS = 30
    MIX_PRECISION = True
    DATA_PATH = "stacked_segmentation/"
    MODEL_SAVE_PATH = "models/unet.pth"

    train_unet()