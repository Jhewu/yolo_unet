import os
import cv2
import torch
from torchvision import transforms, datasets
from tqdm import tqdm

from verifier_net.verifier_net import VerificationNet
from train_unet.dataset import CustomDataset

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
    model = VerificationNet().eval()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

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
            y_pred = torch.nn.functional.sigmoid(model(images))
            
            # **CRITICAL CHANGE:** Get file paths directly from the dataset.
            # This ensures consistency with the data being processed.
            image_path = dataloader.dataset.images[idx]
            label_path = dataloader.dataset.masks[idx]

            if y_pred.item() > 0.5:
                # Use the correct, dataset-provided paths for renaming
                os.rename(image_path, os.path.join(dest_image_dir, os.path.basename(image_path)))
                os.rename(label_path, os.path.join(dest_label_dir, os.path.basename(label_path)))
                print(f"{image_path} Verified to be TRUE positive")
            else:
                print(f"{image_path} Verified to be FALSE positive...")

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