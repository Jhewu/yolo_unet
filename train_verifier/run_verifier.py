import os
import cv2
import torch
from torchvision import transforms, datasets
from tqdm import tqdm

import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image

from ultralytics import YOLO

class CustomDataset(Dataset):
    def __init__(self, root_path, image_path, mask_path, image_size):
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i for i in os.listdir(root_path+f"/{image_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i for i in os.listdir(root_path+f"/{mask_path}/")])

        if len(self.images) != len(self.masks): 
            raise ValueError("Length of images and masks are not the same")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGBA")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def PrepareDataLoader(root_dir, image_dir, label_dir, batch, image_size, workers=4): 
    dataset = CustomDataset(root_dir, image_dir, label_dir, image_size)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=False,
        batch_size=batch, 
        num_workers=workers, 
    )
    return dataloader

def VerifyYOLOCrop():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)

    for split in SPLIT:
        # Define source and destination directories
        dest_image_dir = os.path.join(DEST, "images", split)
        dest_label_dir = os.path.join(DEST, "labels", split)
        CreateDir(dest_image_dir), CreateDir(dest_label_dir)

        # DataLoader will handle sorting and subsampling as defined in CustomDataset
        dataloader = PrepareDataLoader(
            root_dir=DATASET,
            image_dir=os.path.join("images", split),
            label_dir=os.path.join("labels", split),
            batch=1,
            image_size=IMG_SIZE
        )

        for idx, batch in enumerate(tqdm(dataloader)):
            images, _ = batch

            # **CRITICAL CHANGE:** Get file paths directly from the dataset.
            # This ensures consistency with the data being processed.
            image_path = dataloader.dataset.images[idx]
            label_path = dataloader.dataset.masks[idx]

            results = model(image_path)
            prob0, prob1 = results[0].probs.data

            if prob0.item() > prob1.item(): 
                print(f"{image_path} Verified to be FALSE positive...")
            else:
                # Use the correct, dataset-provided paths for renaming
                os.rename(image_path, os.path.join(dest_image_dir, os.path.basename(image_path)))
                os.rename(label_path, os.path.join(dest_label_dir, os.path.basename(label_path)))
                print(f"{image_path} Verified to be TRUE positive")

if __name__ == "__main__": 
    DATASET = "ssa_yolo_cropped_n_good"
    DEST = f"{DATASET}_verified"
    MODEL_PATH = "/home/jun/Desktop/inspirit/yolo_unet/train_verifier/train_yolo12n-cls_finetuned_ssa_best/yolo12n-cls_/home/jun/Desktop/inspirit/yolo_unet/train_verifier/ssa_verifier_dataset_n_good/weights/best.pt"
    SPLIT = ["test", "train", "val"]
    IMG_SIZE = 96

    VerifyYOLOCrop()