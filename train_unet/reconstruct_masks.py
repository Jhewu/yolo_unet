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

def reconstruct_masks(split, root_dest_dir): 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CustomDataset(root_path=DATA_PATH, 
                            image_path=os.path.join("images", split), 
                            mask_path=os.path.join("labels", split), 
                            image_size=UNET_IMG_SIZE)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=False)
    
    model = UNet(in_channels=4, widths=WIDTHS, num_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

    dest_dir = os.path.join(root_dest_dir, split)
    CreateDir(dest_dir)

    ### Counting Images with Metadata Detected
    ### ALL OF THEM SHOULD HAVE METADATA
    total_positive = 0
    total_negative = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(dataloader)):
            img = img_mask[0].float().to(device)
            image_path = dataloader.dataset.images[idx]

            # Open and read Exif Metadata
            pil_img = Image.open(image_path)
            coords = read_metadata(pil_img)

            if coords != None:
                total_positive+=1

                pred_mask = torch.nn.functional.sigmoid(model(img))

                # Resize the predictions
                x1, y1, x2, y2 = coords
                height, width = abs(int(y1)-int(y2)), abs(int(x1)-int(x2))
                transform = transforms.Resize((height, width))
                pred_mask = transform(pred_mask).squeeze(0).squeeze(0)

                # Insert the predictions to full size empty mask
                full_size_mask = torch.zeros(OG_IMG_SIZE, OG_IMG_SIZE, device=device)
                full_size_mask[int(y1):int(y2), int(x1):int(x2)] = pred_mask

                # Binarize the mask
                full_size_mask = (full_size_mask > 0.5).float()

                # Save the full size mask
                dest_image_dir = os.path.join(dest_dir, os.path.basename(image_path))
                cv2.imwrite(dest_image_dir, (full_size_mask.cpu().numpy() * 255).astype(np.uint8))
            else:
                total_negative+=1

                dest_image_dir = os.path.join(dest_dir, os.path.basename(image_path))
                full_size_mask = torch.zeros(OG_IMG_SIZE, OG_IMG_SIZE, device=device)
                cv2.imwrite(dest_image_dir, (full_size_mask.cpu().numpy() * 255).astype(np.uint8))

    print("\nTotal images with metadata: ", total_positive)
    print("Total images without metadata: ", total_negative)

    if total_negative > 1: 
        print(f"\nWARNING: Metadata not present in {total_negative} images")

if __name__ == "__main__": 
    OG_IMG_SIZE = 192
    UNET_IMG_SIZE = 128
    DATA_PATH = "yolo_cropped_verified"
    WIDTHS = [32, 64, 128, 256, 512]
    MODEL_PATH = "/home/jun/Desktop/inspirit/yolo_unet/train_unet/runs/unet_2025_08_23_20_26_07/weights/best.pth"
    SPLIT = "train"
    DEST_DIR = f"reconstructed_{SPLIT}/labels"
    
    reconstruct_masks(SPLIT, DEST_DIR)