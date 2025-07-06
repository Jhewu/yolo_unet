"""
A preprocessing utility that will use a given image
as input, binarize it, obtain the center points, and from
the center point crop the image by a given image size

Essentially it crops into the brain subject
"""

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

MODALITY = ["t1c", "t1n", "t2f" ,"t2w"]  
IMAGE_SIZE = 192

"""
The code below is borrowed from Farah Alarbid
Credits here: https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format
"""

def ProcessMask(image_path, destination, threshold=10, color=False):
    if color:
        pass
        print("WARNING: COLOR NOT IMPLEMENTED YET")
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Image processing
    _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_image = image * binary_mask

    max_area = 0
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if max_area < (width*height):
            max_area = (width*height)
            x_center = (x + width // 2)
            y_center = (y + height // 2)

    cropped_image = CropCenter(cleaned_image, IMAGE_SIZE, x_center, y_center)
    
    # -------------------------------------
    # LEAVE FOR TROUBLESHOOTING

    # cv2.imshow("original", image)
    # cv2.imshow("cropped", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # -------------------------------------

    cropped_image_path = os.path.join(destination, os.path.basename(image_path))
    cv2.imwrite(cropped_image_path, cropped_image)

def CropCenter(image, crop_size, x_center, y_center, pad_value=0):
    """
    Crops an image from the center with a specified size. If the desired crop
    exceeds the image boundaries, padding is added.

    Args:
        image (np.ndarray): The input image (height, width, channels) or (height, width).
        crop_size (int): The desired side length of the square crop.
        x_center (int): The x-coordinate (width) of the center of the crop.
        y_center (int): The y-coordinate (height) of the center of the crop.
        pad_value (int or tuple): The value to use for padding. For grayscale images,
                                   an int (e.g., 0 for black). For color images,
                                   a tuple (e.g., (0, 0, 0) for black).

    Returns:
        np.ndarray: The cropped and potentially padded image.
    """
    height, width = image.shape[0], image.shape[1]
    half_crop_size = crop_size // 2

    # Calculate the desired crop boundaries (each corner)
    x1 = x_center - half_crop_size
    y1 = y_center - half_crop_size
    x2 = x_center + half_crop_size
    y2 = y_center + half_crop_size

    # Calculate crop boundaries within the image
    x1_bound = max(0, x1)
    y1_bound = max(0, y1)
    x2_bound = min(width, x2)
    y2_bound = min(height, y2)

    # Calculate padding amounts
    pad_left = abs(min(0, x1))
    pad_top = abs(min(0, y1))
    pad_right = abs(min(0, width - x2))
    pad_bottom = abs(min(0, height - y2))

    # Perform the initial crop
    cropped_image = image[y1_bound:y2_bound, x1_bound:x2_bound]

    # Apply padding if necessary
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        # Determine the number of channels for padding
        if image.ndim == 3:  # Color image
            if not isinstance(pad_value, tuple):
                raise ValueError("For color images, 'pad_value' must be a tuple (R, G, B).")
            # Pad for each channel
            padding_widths = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        elif image.ndim == 2:  # Grayscale image
            if not isinstance(pad_value, (int, float)):
                raise ValueError("For grayscale images, 'pad_value' must be an int or float.")
            padding_widths = ((pad_top, pad_bottom), (pad_left, pad_right))
        else:
            raise ValueError("Unsupported image dimensions. Expected 2 (grayscale) or 3 (color).")

        cropped_image = np.pad(cropped_image, padding_widths, mode='constant', constant_values=pad_value)

    return cropped_image

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Main():
    for mod in MODALITY:
        input_folder = f"{mod}/images"
        DEST_FOLDER = f"{mod}_cropped/images"

        input_folder = os.path.join(os.getcwd(), input_folder)
        dataset_split = os.listdir(input_folder)

        max_workers = 10 # adjust based on your syster's capabilities

        # ---------------------------------------------------------------
        # LEAVE FOR TROUBLESHOOTING

        # for split in dataset_split: 
        #     split_path = os.path.join(input_folder, split)
        #     image_list = os.listdir(split_path)

        #     dest_dir = os.path.join(DEST_FOLDER, split)
        #     CreateDir(dest_dir)
            
        #     for image in image_list[:2]: 
        #         image_path = os.path.join(split_path, image)
        #         ProcessMask(image_path, dest_dir)

        # ---------------------------------------------------------------

        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            for split in dataset_split: 
                split_path = os.path.join(input_folder, split)
                image_list = os.listdir(split_path)

                dest_dir = os.path.join(DEST_FOLDER, split)
                CreateDir(dest_dir)
            
                for image in image_list: 
                    image_path = os.path.join(split_path, image)
                    executor.submit(ProcessMask, image_path, dest_dir)

if __name__ == "__main__":
    Main()
    print("\nFinish processing images, check your directory\n")