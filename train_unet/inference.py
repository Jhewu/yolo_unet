import torch
import matplotlib.pyplot as plt
from PIL import Image
import random

from dataset import CustomDataset
from unet import UNet

def pred_show_image_grid(num_batches=3, split="val"):
    model = UNet(in_channels=4, widths=WIDTHS, num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.eval()

    image_dataset = CustomDataset(DATA_PATH, f"images/{split}", f"labels/{split}", IMAGE_SIZE)
    
    # Store processed images, original masks, and predicted masks for all batches
    all_images = []
    all_orig_masks = []
    all_pred_masks = []
    all_masked_images = []

    # Loop through the desired number of batches
    for i in range(min(num_batches, len(image_dataset))): # Ensure we don't go out of bounds
        seed = random.randint(0, len(image_dataset)-num_batches)

        img, orig_mask = image_dataset[i+seed] 

        img_input = img.float().to(DEVICE).unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            pred_mask = model(img_input)
            pred_mask = torch.sigmoid(pred_mask)

        # Process and prepare for display
        all_images.append(img.permute(1, 2, 0).cpu().detach())
        
        processed_pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        processed_pred_mask = (processed_pred_mask > 0.5).float()

        all_pred_masks.append(processed_pred_mask)
        all_orig_masks.append(orig_mask.permute(1, 2, 0).cpu().detach())

        # Multiply the mask and the original image
        masked_img = img.squeeze(0).cpu().detach().permute(1, 2, 0) * processed_pred_mask

        all_masked_images.append(masked_img)

    # Plotting
    fig, axes = plt.subplots(4, num_batches, figsize=(4 * num_batches, 12))
    if num_batches == 1: # Handle case of single batch to avoid indexing error on axes
        axes = axes.reshape(-1, 1) # Make it 2D for consistent indexing
    
    titles = ["Original Image", "Original Mask", "Predicted Mask"]

    for i in range(num_batches):
        # Original Image
        axes[0, i].imshow(all_images[i].squeeze())
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis('off')

        # Original Mask
        axes[1, i].imshow(all_orig_masks[i].squeeze(), cmap="gray")
        axes[1, i].set_title(f"Original Mask {i+1}")
        axes[1, i].axis('off')

        # Predicted Mask
        axes[2, i].imshow(all_pred_masks[i].squeeze(), cmap="gray")
        axes[2, i].set_title(f"Predicted Mask {i+1}")
        axes[2, i].axis('off')

        # Masked Image
        axes[3, i].imshow(all_masked_images[i].squeeze())
        axes[3, i].set_title(f"Mask Images {i+1}")
        axes[3, i].axis('off')
            
    plt.tight_layout()
    plt.show()

def single_image_inference():
    model = UNet(in_channels=4, widths=WIDTHS, num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))

    dataset = CustomDataset(DATA_PATH, "images/train", "labels/train", IMAGE_SIZE)

    ### COMMENT: Open image from path and then perform transform on the image
    # Lastly, convert the image into floating point and move to the device
    img = dataset.transform(Image.open(SINGLE_IMAGE)).float().to(DEVICE)
    img = img.unsqueeze(0) # --> COMMENT: Adds the batch dimension
   
    pred_mask = model(img)
    pred_mask = torch.sigmoid(pred_mask)

    ### COMMENT: Batch dimension is removed, and tensor is returned to CPU
    # detach() creates a new tensor that does not track gradients
    # tensor dimensions are rearranged from (channels, height, width)
    # to standard images
    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    ### COMMENT: Same as above, but also binarizes the masks
    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask = (pred_mask > 0.5).float()

    fig = plt.figure()
    for i in range(1, 5): 
        fig.add_subplot(1, 4, i)
        if i == 1:
            plt.imshow(img)
        elif i == 2:
            plt.imshow(pred_mask, cmap="gray")
        elif i == 3:
            mask = Image.open(SINGLE_IMAGE_LABEL).convert("L")
            plt.imshow(mask, cmap="gray")
        else: 
            masked_img = img * pred_mask
            plt.imshow(masked_img)

    plt.show()

if __name__ == "__main__":
    DATA_PATH = "yolo_cropped"

    SINGLE_IMAGE_LABEL = "BraTS-SSA-00041-00036-t1c.png"
    SINGLE_IMAGE = "yolo_cropped/images/test/BraTS-SSA-00041-00036-t1c.png"

    MODEL_PATH = "runs/unet_2025_08_11_15_09_46/weights/best.pth"
    WIDTHS = [32, 64, 128, 256]
    IMAGE_SIZE = 128

    torch.cuda.is_available = lambda : False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    pred_show_image_grid(10)
    # single_image_inference()