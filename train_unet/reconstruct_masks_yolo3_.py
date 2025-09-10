import os
import cv2
import torch
import piexif
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from ultralytics import YOLO
from dataset import CustomDataset
from custom_predictor.custom_detection_predictor import CustomDetectionPredictor, CustomSegmentationPredictor

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

    images = sorted([os.path.join(DATA_PATH, "images", split, i) for i in os.listdir(os.path.join(DATA_PATH, "images", split))])
    
    args = dict(conf=0.7, save=False, device="cuda", imgsz=UNET_IMG_SIZE, batch=1, verbose=False)  
    predictor = CustomSegmentationPredictor(overrides=args)
    predictor.setup_model(MODEL_PATH)

    dest_dir = os.path.join(root_dest_dir, split)
    CreateDir(dest_dir)

    ### Counting Images with Metadata Detected
    ### ALL OF THEM SHOULD HAVE METADATA
    total_positive = 0
    total_negative = 0

    for idx, image_path in enumerate(tqdm(images[:100])):
        # Open and read Exif Metadata
        # pil_img = Image.open(image_path)
        # coords = read_metadata(pil_img)

        # if coords != None:
            total_positive+=1

            results = predictor(image_path)
            pred_mask = results[0].masks  # Masks object for segmentation masks outputs

            # print(results[0].boxes.coeffs)
            # print(pred_mask)

            if pred_mask != None: 
            #     # Resize the predictions
            #     # x1, y1, x2, y2 = coords
            #     height, width = abs(int(y1)-int(y2)), abs(int(x1)-int(x2))
            #     transform = transforms.Resize((height, width))
                if pred_mask.shape[0] > 1: 
                    pred_mask = pred_mask.data.squeeze(0)[0]
            #         print(pred_mask)
                else: pred_mask = pred_mask.data.squeeze(0)

                print(np.unique(pred_mask.cpu().numpy()))

                # Insert the predictions to full size empty mask
                # full_size_mask = torch.zeros(OG_IMG_SIZE, OG_IMG_SIZE, device=device)
                # full_size_mask[int(y1):int(y2), int(x1):int(x2)] = pred_mask

                # print(pred_mask.data.unsqueeze(0)[0].cpu().numpy().squeeze(0).shape)

                # Save the full size mask
                dest_image_dir = os.path.join(dest_dir, os.path.basename(image_path))
                # cv2.imwrite(dest_image_dir, pred_mask.cpu().numpy()*255)
        
        # else:
        #     total_negative+=1

        #     dest_image_dir = os.path.join(dest_dir, os.path.basename(image_path))
        #     full_size_mask = torch.zeros(OG_IMG_SIZE, OG_IMG_SIZE, device=device)
        #     cv2.imwrite(dest_image_dir, (full_size_mask.cpu().numpy() * 255).astype(np.uint8))

    print("\nTotal images with metadata: ", total_positive)
    print("Total images without metadata: ", total_negative)

    if total_negative > 1: 
        print(f"\nWARNING: Metadata not present in {total_negative} images")

if __name__ == "__main__": 
    OG_IMG_SIZE = 192
    UNET_IMG_SIZE = 192
    DATA_PATH = "stacked_segmentation"
    MODEL_PATH = "/home/jun/Desktop/inspirit/yolo_unet/train_unet/train_yolo12n-seg_2025_08_27_01_01_59/yolo12n-seg_data/weights/best.pt"
    SPLIT = "test"
    DEST_DIR = f"reconstructed_{SPLIT}/labels"
    
    reconstruct_masks(SPLIT, DEST_DIR)