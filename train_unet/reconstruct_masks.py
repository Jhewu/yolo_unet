import os
import cv2
import torch
import piexif
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from unet import UNet
from dataset import CustomDataset

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name) 

def read_metadata(img):
    exif = img.getexif()
    exif_bytes = exif.tobytes()
    exif_dict= piexif.load(exif_bytes)

    # Grab the raw bytes of the UserComment tag
    raw_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)

    if raw_comment is None:
        return None

    # Convert the tuple (or bytes) to a real string
    # The EXIF spec says the first 8 bytes are an encoding prefix.
    # If you wrote the string yourself (without a prefix) it will
    # simply be the raw UTF‑8 bytes, so we can decode directly.
    comment = bytes(raw_comment).decode("utf-8", errors="ignore")

    return comment.split(",")

def reconstruct_masks(split, dest_dir): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CustomDataset(DATA_PATH, f"images/{split}", f"labels/{split}", IMAGE_SIZE)

    image_paths = sorted([DATA_PATH+f"/images/{split}/"+i for i in os.listdir(DATA_PATH+f"/images/{split}/")])

    dataloader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False)
    
    model = UNet(in_channels=4, widths=WIDTHS, num_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

    dest_dir = os.path.join(dest_dir, split)
    CreateDir(dest_dir)

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(dataloader)):
            img = img_mask[0].float().to(device)

            # Open and read Exif Metadata
            pil_img = Image.open(image_paths[idx])
            x1, y1, x2, y2 = read_metadata(pil_img)
            pred_mask = model(img)

            # Resize the predictions
            height, width = abs(int(y1)-int(y2)), abs(int(x1)-int(x2))
            transform = transforms.Resize((height, width))
            pred_mask = transform(pred_mask)

            pred_mask = torch.sigmoid(pred_mask)

            # Insert the predictions to full size empty mask
            full_size_mask = torch.zeros(IMAGE_SIZE, IMAGE_SIZE,device=device)
            full_size_mask[int(y1):int(y2), int(x1):int(x2)] = pred_mask

            # Binarize the mask
            full_size_mask = full_size_mask.squeeze(0).squeeze(0)
            full_size_mask = (full_size_mask > 0.5).float()

            # Save the full size mask
            dest_image_dir = os.path.join(dest_dir, os.path.basename(image_paths[idx]))
            cv2.imwrite(dest_image_dir, (full_size_mask.cpu().numpy() * 255).astype(np.uint8))

            # cv2.imwrite(dest_image_dir, full_size_mask.numpy())

if __name__ == "__main__": 
    IMAGE_SIZE = 192
    DATA_PATH = "ground_truth_cropped_all"
    WIDTHS = [32, 64, 128, 256]
    MODEL_PATH = "/home/jun/Desktop/inspirit/yolo_unet/train_unet/runs/unet_2025_08_20_15_51_45/weights/best.pth"
    SPLIT = "test"
    DEST_DIR = f"reconstructed_{SPLIT}/labels"
    
    reconstruct_masks(SPLIT, DEST_DIR)