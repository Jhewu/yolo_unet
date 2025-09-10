import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Set up a basic logger for a clean output
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_masks(input_dir, output_dir):
    """
    Processes all images in a directory, converts 255-pixel values to 1, and saves them.
    This ensures binary masks are correctly formatted for the MaskToPolygons2 logic.
    """
    CreateDir(output_dir)

    # Get a list of all mask image files to process
    mask_paths = [p for p in Path(input_dir).iterdir() if p.suffix in {'.png', '.jpg', '.jpeg'}]

    if not mask_paths:
        LOGGER.info(f"No mask images found in {input_dir}. Skipping.")
        return

    # Use tqdm to show a progress bar
    for mask_path in tqdm(mask_paths, desc=f"Converting masks in {input_dir}"):
        # Read the mask image in grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            LOGGER.warning(f"Could not read image {mask_path}. Skipping.")
            continue

        # Create a new binary mask. All pixels with a value > 0 (i.e., 255)
        # will be set to 1. All others (0) will remain 0.
        processed_mask = (mask > 0).astype(np.uint8) * 1
        
        # Note: The original function you provided expects class values of 1, 2, 3, etc.
        # This conversion makes all non-zero pixels into the value 1,
        # which is what your `pixel_to_class_mapping` for a single class expects:
        # pixel_to_class_mapping = {1: 0}

        # Construct the output path and save the processed mask.
        output_path = Path(output_dir) / mask_path.name
        cv2.imwrite(str(output_path), processed_mask)

        LOGGER.info(f"Processed and stored at {output_path}")

if __name__ == "__main__": 

    for split in ["test", "train", "val"]:
        label_dir = os.path.join("gli_stacked_segmentation", "masks", split)
        dest_dir = os.path.join("gli_stacked_segmentation", "mask2", split)
        convert_masks(label_dir, dest_dir)