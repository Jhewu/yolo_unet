import torch
from unet import UNet
import numpy as np

if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    WIDTHS = [2, 4, 6, 8]

    model = UNet(in_channels=3, widths=WIDTHS, num_classes=1).to(device)

    image = torch.zeros((192, 192, 3))

    img_input = image.float().to(device).unsqueeze(0) # Add batch dimension

    pred = model(image)

    print(pred)