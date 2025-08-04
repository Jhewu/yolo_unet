import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from dataset import CustomDataset
from unet import UNet

torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pred_show_image_grid(data_path, model_pth, device, num_batches=3):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()

    image_dataset = CustomDataset(data_path, "train", "train_masks", 128)
    
    # Store processed images, original masks, and predicted masks for all batches
    all_images = []
    all_orig_masks = []
    all_pred_masks = []

    # Loop through the desired number of batches
    for i in range(min(num_batches, len(image_dataset))): # Ensure we don't go out of bounds
        img, orig_mask = image_dataset[i] 

        img_input = img.float().to(device).unsqueeze(0) # Add batch dimension

        with torch.no_grad():
            pred_mask = model(img_input)

        # Process and prepare for display
        all_images.append(img.permute(1, 2, 0).cpu().detach())
        
        processed_pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1, 2, 0)
        processed_pred_mask[processed_pred_mask < 0] = 0
        processed_pred_mask[processed_pred_mask > 0] = 1

        all_pred_masks.append(processed_pred_mask)
        all_orig_masks.append(orig_mask.permute(1, 2, 0).cpu().detach())

        # Multiply the mask and the original image
        # prediction = img * pred_mask
        # original = img * orig_mask

        # all_pred_masks.append(prediction)
        # all_orig_masks.append(original)

    # Plotting
    fig, axes = plt.subplots(3, num_batches, figsize=(4 * num_batches, 12))
    if num_batches == 1: # Handle case of single batch to avoid indexing error on axes
        axes = axes.reshape(-1, 1) # Make it 2D for consistent indexing
    
    titles = ["Original Image", "Original Mask", "Predicted Mask"]

    for i in range(num_batches):
        # Original Image
        axes[0, i].imshow(all_images[i].squeeze(), cmap="gray")
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
            
    plt.tight_layout()
    plt.show()

# def single_image_inference(image_pth, model_pth, device):
#     model = UNet(in_channels=3, num_classes=1).to(device)
#     model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()])

#     img = transform(Image.open(image_pth)).float().to(device)
#     img = img.unsqueeze(0)
   
#     pred_mask = model(img)

#     img = img.squeeze(0).cpu().detach()
#     img = img.permute(1, 2, 0)

#     pred_mask = pred_mask.squeeze(0).cpu().detach()
#     pred_mask = pred_mask.permute(1, 2, 0)
#     pred_mask[pred_mask < 0]=0
#     pred_mask[pred_mask > 0]=1

#     fig = plt.figure()
#     for i in range(1, 3): 
#         fig.add_subplot(1, 2, i)
#         if i == 1:
#             plt.imshow(img, cmap="gray")
#         else:
#             plt.imshow(pred_mask, cmap="gray")
#     plt.show()

if __name__ == "__main__":
    SINGLE_IMG_PATH = "./data/train/BraTS-PED-00021-00059-t1c.png"
    DATA_PATH = "data"
    MODEL_PATH = "models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device, 25)
    # single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)