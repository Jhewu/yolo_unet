from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import numpy as np
import argparse
import os
import cv2

def crop_center(image, crop_size, x_center, y_center, pad_value=(0, 0, 0, 0)):
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
    # if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
    #     padding_widths = ((pad_top, pad_bottom), (pad_left, pad_right))

    #     if image.ndim == 3:  # Color image (including 3-channel and 4-channel)
    #         if not isinstance(pad_value, tuple) or len(pad_value) != image.shape[2]:
    #             raise ValueError(f"For color images, 'pad_value' must be a tuple with {image.shape[2]} elements.")
    #         padding_widths += ((0, 0),) # Add padding for channel dimension (no padding)
    #     elif image.ndim == 2:  # Grayscale image
    #         if not isinstance(pad_value, (int, float)):
    #             raise ValueError("For grayscale images, 'pad_value' must be an int or float.")
    #     else:
    #         raise ValueError(f"Unsupported image dimensions. Expected 2 (grayscale) or 3 (color with any number of channels), but got {image.ndim}.")

    #     cropped_image = np.pad(cropped_image, padding_widths, mode='constant', constant_values=pad_value)

    # return cropped_image
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:

        # Determine the number of channels for padding
        if image.ndim == 3:  # Color image
            # if not isinstance(pad_value, tuple):
            #     raise ValueError("For color images, 'pad_value' must be a tuple (R, G, B, A).")
            # Pad for each channel
            padding_widths = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            print("\nThis is padding widths ", padding_widths)
            print("\nThis is image shape ", image.shape)

        elif image.ndim == 2:  # Grayscale image
            # if not isinstance(pad_value, (int, float)):
            #     raise ValueError("For grayscale images, 'pad_value' must be an int or float.")
            padding_widths = ((pad_top, pad_bottom), (pad_left, pad_right)) 
        cropped_image = np.pad(cropped_image, padding_widths, mode='constant', constant_values=0)

    return cropped_image

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def crop_from_yolo(results, label_paths, image_dest_dir, label_dest_dir): 

    # check if there's any objects with a prediction, and obtain such coordinates
    coords = None
    for result in results: 
        boxes = result.boxes
        if len(boxes) > 0: 
            coords = boxes.xywh[0]
            break
        

        """
        VERIFY IF ITS WORKING CORRECTLY, PARTICULARLY AT THE ELSE CONDITION IF THERE'S NO DETECTION
        
        """
    
    # process images based on whether a prediction was found
    if coords is not None:
        center_x = int(coords[0])
        center_y = int(coords[1])
        """
        VERIFY IF ITS WORKING CORRECTLY, PARTICULARLY AT THE ELSE CONDITION IF THERE'S NO DETECTION
        """
        cropped_images = [crop_center(cv2.imread(path, cv2.IMREAD_UNCHANGED), CROP_SIZE, center_x, center_y) for path in image_paths]
        cropped_labels = [crop_center(cv2.imread(path, cv2.IMREAD_UNCHANGED), CROP_SIZE, center_x, center_y) for path in label_paths]
    else:
        cropped_images = [crop_center(cv2.imread(path, cv2.IMREAD_UNCHANGED), CROP_SIZE, col//2, row//2) for path in image_paths]
        cropped_labels = [crop_center(cv2.imread(path, cv2.IMREAD_UNCHANGED), CROP_SIZE, col//2, row//2) for path in label_paths]

    for i, image in enumerate(cropped_images):
        image_basename = os.path.basename(image_paths[i])
        label_basename = os.path.basename(label_paths[i])
        
        image_path = os.path.join(image_dest_dir, image_basename)
        label_path = os.path.join(label_dest_dir, label_basename)

        # print(image_path)
        # print(label_path)
        # cv2.imwrite(image_path, image)
        # cv2.imwrite(label_path, cropped_labels[i])

def yolo_crop_async(): 
    image_dir = os.path.join(IN_DIR, "images")
    label_dir = os.path.join(IN_DIR, "labels")

    image_dest_dir = os.path.join(OUT_DIR, "images")
    label_dest_dir = os.path.join(OUT_DIR, "labels")

    dataset_split = ["test", "train", "val"]

    for split in dataset_split:
        image_split_dir = os.path.join(image_dir, split)
        label_split_dir = os.path.join(label_dir, split) 

        image_split_dest_dir = os.path.join(image_dest_dir, split)
        label_split_dest_dir = os.path.join(label_dest_dir, split)

        image_list = os.listdir(image_split_dir)
        label_list = os.listdir(label_split_dir)

        # Ensure image matches label
        image_list.sort(), label_list.sort()

        # Construct the full directories of images and labels
        image_full_paths = [os.path.join(image_split_dir, image) for image in image_list]
        label_full_paths = [os.path.join(label_split_dir, image) for image in label_list]
        create_dir(image_split_dest_dir), create_dir(label_split_dest_dir)

        model = YOLO(MODEL_DIR)

        image = cv2.imread(image_full_paths[0], cv2.IMREAD_UNCHANGED)
        row, col, channel = image.shape

        # Batch the directories
        for i in range(0, len(image_full_paths), BATCH_SIZE): 
            # Create the batches
            image_paths = image_full_paths[i:i+BATCH_SIZE]
            label_paths = label_full_paths[i:i+BATCH_SIZE]

            # Perform inference
            image_results = model(image_paths, conf=CONFIDENCE)

            """IMPLEMENT THE FILTERING BLOCK HERE OR SOMEWHERE ELSE, YOLO AUTOMATICALLY DISCARD OR GATES IMAGES"""
            crop_from_yolo(image_results, image_paths, label_paths, image_split_dest_dir, label_split_dest_dir)

            # Crop the images from YOLO coordinates (labels use the same crop coordinates as image)
            # with ThreadPoolExecutor(max_workers=WORKERS) as executor: 
            #     executor.submit(crop_from_yolo, image_results, image_paths, label_paths, image_split_dest_dir, label_split_dest_dir)

if __name__ == "__main__": 
    # ---------------------------------------------------
    des="""
    Performs YOLO cropping on a preprocessed BraTS 2D
    dataset, to prepare them for U-NET like segmentation
    training 
    """
    # ---------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument("--model_dir", type=str,help='YOLO model directory\t[None]')

    parser.add_argument('--confidence', type=int, help='confidence for binarizing the image\t[15]')
    parser.add_argument('--crop_size', type=int, help='final NxN image crop\t[64]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    parser.add_argument('--use_cpu', action='store_true', help='Use cpu for inference (YOLO on multiple threads), if not set, it defaults to GPU')
    parser.add_argument('--batch_size', type=int, help='batch size used to process YOLO, depending on your GPU capabilities\t[64]')

    parser.add_argument('--filter', action='store_true', help='Enable YOLO Gating, discard images under the confidence score')

    args = parser.parse_args()

    """REORGANIZE THESE IN THE FUTURE"""
    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "stacked_segmentation"
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "yolo_cropped"
    if args.model_dir is not None:
        MODEL_DIR = args.model_dir
    else: MODEL_DIR = "yolo_weights/best.pt"
    if args.crop_size is not None:
        CROP_SIZE = args.crop_size
    else: CROP_SIZE = 176
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    else: BATCH_SIZE = 1024
    if args.confidence is not None:
        CONFIDENCE = args.confidence
    else: CONFIDENCE = 0.70
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.filter is not None:
        FILTER = args.filter
    else: FILTER = False
    if args.use_cpu is not None:
        USE_CPU = args.use_cpu
    else: USE_CPU = False

    yolo_crop_async()
