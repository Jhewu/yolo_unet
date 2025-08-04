import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_path, image_path, mask_path, image_size, isnpy=False):
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i for i in os.listdir(root_path+f"/{image_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i for i in os.listdir(root_path+f"/{mask_path}/")])

        if len(self.images) != len(self.masks): 
            raise ValueError("Length of images and masks are not the same")
        
        # SUBSAMPLE IMPLEMENTED HERE
        self.images = self.images[:int(len(self.images)*0.5)]
        self.masks = self.masks[:int(len(self.masks)*0.5)]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5, ))])
        
        self.isnpy = isnpy

    def __getitem__(self, index):
        if self.isnpy: 
            img = np.load(self.images[index])
            img = np.load(self.masks[index])
        else: 
            img = Image.open(self.images[index]).convert("RGBA")
            mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)

