import os
from random import shuffle as randshuffle
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_path, image_path, mask_path, image_size):
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i for i in os.listdir(root_path+f"/{image_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i for i in os.listdir(root_path+f"/{mask_path}/")])

        print(self.images)
        print(self.masks)

        # Check if the sorted list 
        if len(self.images) != len(self.masks): 
            raise ValueError("Length of images and masks are not the same")
        
        # # Create a random list of indices and shuffle
        # indices = list(range(len(self.images)))
        # randshuffle(indices)

        # # Randomly sort both lists using the shuffled indices
        # self.images = [self.images[i] for i in indices]
        # self.masks = [self.masks[i] for i in indices]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
