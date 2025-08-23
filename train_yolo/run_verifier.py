import os
import cv2
import torch
from torchvision import transforms, datasets
from tqdm import tqdm

# from ..train_unet.dataset import CustomDataset
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image

from ultralytics import YOLO

class CustomDataset(Dataset):
    def __init__(self, root_path, image_path, mask_path, image_size, subsample=1.0, isnpy=False):
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i for i in os.listdir(root_path+f"/{image_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i for i in os.listdir(root_path+f"/{mask_path}/")])

        if len(self.images) != len(self.masks): 
            raise ValueError("Length of images and masks are not the same")
        
        # SUBSAMPLE IMPLEMENTED HERE
        self.images = self.images[:int(len(self.images)*subsample)]
        self.masks = self.masks[:int(len(self.masks)*subsample)]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        
        self.isnpy = isnpy

    def __getitem__(self, index):
        if self.isnpy: 
            img = np.load(self.images[index])
            mask = np.load(self.masks[index])
        else: 
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
    model = YOLO("train_yolo12s-cls_2025_08_23_16_02_39/yolo12s-cls_/home/jun/Desktop/inspirit/yolo_unet/train_yolo/verifier_dataset/weights/best.pt")
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

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

# def VerifyYOLOCrop():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = VerificationNet().eval()
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

#     for split in SPLIT: 
#         images = sorted([os.path.join(DATASET, "images", split, i) for i in os.listdir(os.path.join(DATASET, "images", split))])
#         labels = sorted([os.path.join(DATASET, "labels", split, i) for i in os.listdir(os.path.join(DATASET, "labels", split))])

#         dest_image_dir = os.path.join(DEST, "images", split)
#         dest_label_dir = os.path.join(DEST, "labels", split)
#         CreateDir(dest_image_dir), CreateDir(dest_label_dir)

#         dataloader = PrepareDataLoader(DATASET, os.path.join("images", split), os.path.join("labels", split), 1, IMG_SIZE)

#         for idx, batch in enumerate(tqdm(dataloader)): 
#             images, _ = batch  
#             y_pred = torch.nn.functional.sigmoid(model(images))

#             if y_pred.item() > 0.5: 
#                 os.rename(images[idx], os.path.join(dest_image_dir, os.path.basename(images[idx])))
#                 os.rename(labels[idx], os.path.join(dest_label_dir, os.path.basename(labels[idx])))
#         #         # cv2.imwrite(os.path.join(dest_image_dir, os.path.basename(image_path)), cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
#         #         # cv2.imwrite(os.path.join(dest_label_dir, os.path.basename(label_path)), cv2.imread(label_path, cv2.IMREAD_UNCHANGED))
#                 print(f"{images[idx]} Verified to be TRUE positive")
#             else: print(f"{images[idx]} Verified to be FALSE positive...")

if __name__ == "__main__": 
    DEST = "yolo_cropped_verified"
    MODEL_PATH = "/home/jun/Desktop/inspirit/yolo_unet/runs/unet_2025_08_23_14_09_02/weights/best.pth"
    DATASET = "yolo_cropped"
    SPLIT = ["test", "train", "val"]
    IMG_SIZE = 224

    VerifyYOLOCrop()