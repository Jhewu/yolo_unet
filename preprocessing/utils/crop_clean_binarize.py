import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse

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

def ProcessMask(image_path, label_path, image_dest, label_dest, threshold=10, color=False):
    """
    The code below is borrowed from Farah Alarbid
    Credits here: https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format
    """

    if color:
        pass
        print("WARNING: COLOR NOT IMPLEMENTED YET")
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    row, col = image.shape

    if np.all(image == 0): 
        """
        NOTE TO FUTURE SELF: 
        Currently, this script contains a flaw, and it will skip
        all images containing "0" thus creating an uneven amount 
        of images across modality, making the stacking function
        not work correctly. The resizing aims to fix this issue
        """

        cropped_image = CropCenter(image, CROP_SIZE, col//2, row//2)
        cropped_label = CropCenter(label, CROP_SIZE, col//2, row//2)

        if SEGMENTATION: 
            # rebinarize to the range of [0, 1, 255], as opposed to [0, 255]
            # such that it can work with the convert to segment function from YOLO
            cropped_label = (cropped_label / 255).astype(np.uint8)

    else: 
        # Image processing
        _, image_binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        _, label_binary_mask = cv2.threshold(label, 250, 255, cv2.THRESH_BINARY)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image_binary_mask = cv2.morphologyEx(image_binary_mask, cv2.MORPH_CLOSE, kernel)
        image_binary_mask = cv2.morphologyEx(image_binary_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(image_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_image = image * image_binary_mask

        max_area = 0
        if len(contours) == 0: 
            cropped_image = CropCenter(image, CROP_SIZE, col//2, row//2)
            cropped_label = CropCenter(label, CROP_SIZE, col//2, row//2)

            if SEGMENTATION: 
                # rebinarize to the range of [0, 1, 255], as opposed to [0, 255]
                # such that it can work with the convert to segment function from YOLO
                cropped_label = (cropped_label / 255).astype(np.uint8)
        
        else: 
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if max_area < (width*height):
                    max_area = (width*height)
                    x_center = (x + width // 2)
                    y_center = (y + height // 2)

            cropped_image = CropCenter(cleaned_image, CROP_SIZE, x_center, y_center)
            cropped_label = CropCenter(label_binary_mask, CROP_SIZE, x_center, y_center)

            if SEGMENTATION: 
                # rebinarize to the range of [0, 1, 255], as opposed to [0, 255]
                # such that it can work with the convert to segment function from YOLO
                cropped_label = (cropped_label / 255).astype(np.uint8)
            
            # -------------------------------------
            # LEAVE FOR TROUBLESHOOTING

            # cv2.imshow("original", image)
            # cv2.imshow("cropped", cropped_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # -------------------------------------

    cropped_image_path = os.path.join(image_dest, os.path.basename(image_path))
    cropped_label_path = os.path.join(label_dest, os.path.basename(label_path))

    cv2.imwrite(cropped_image_path, cropped_image)
    cv2.imwrite(cropped_label_path, cropped_label)

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Main():
    for mod in MODALITY:
        image_dir = f"{IN_DIR}/{mod}/images"
        label_dir = f"{IN_DIR}/{mod}/labels"

        dest_image_dir = f"{OUT_DIR}/{mod}_segmentation/images"
        dest_label_dir = f"{OUT_DIR}/{mod}_segmentation/labels"

        # This directory contains the train, val, test split
        image_dir = os.path.join(os.getcwd(), image_dir)
        dataset_split = os.listdir(image_dir)
    
        # ---------------------------------------------------------------
        # LEAVE FOR TROUBLESHOOTING

        # for split in dataset_split: 
        #     print("\nThis is image dir", dest_image_dir)

        #     # create the directories where images and labels will be accessed
        #     image_split_path = os.path.join(image_dir, split)
        #     label_split_path = os.path.join(label_dir, split)
            
        #     # list of the images and the labels
        #     image_list = os.listdir(image_split_path)
        #     label_list = os.listdir(label_split_path)

        #     # create the destination directories for both images and labels
        #     dest_image_split_dir = os.path.join(dest_image_dir, split)
        #     dest_label_split_dir = os.path.join(dest_label_dir, split)
        #     print(dest_image_split_dir)
        #     print(dest_label_split_dir)
        #     CreateDir(dest_image_split_dir), CreateDir(dest_label_split_dir)

        #     # sort the image and labels to use the same pairs
        #     image_list.sort()
        #     label_list.sort()

        #     # iterate through the indexes
        #     for i in range( len(image_list)):
        #         image_path = os.path.join(image_split_path, image_list[i])
        #         label_path = os.path.join(label_split_path, label_list[i])
        #         ProcessMask(image_path, label_path, dest_image_split_dir, dest_label_split_dir)

        # ---------------------------------------------------------------

        with ThreadPoolExecutor(max_workers=WORKERS) as executor: 
            for split in dataset_split: 
                # create the directories where images and labels will be accessed
                image_split_path = os.path.join(image_dir, split)
                label_split_path = os.path.join(label_dir, split)
                
                # list of the images and the labels
                image_list = os.listdir(image_split_path)
                label_list = os.listdir(label_split_path)

                # create the destination directories for both images and labels
                dest_image_split_dir = os.path.join(dest_image_dir, split)
                dest_label_split_dir = os.path.join(dest_label_dir, split)
                print(dest_image_split_dir)
                print(dest_label_split_dir)
                CreateDir(dest_image_split_dir), CreateDir(dest_label_split_dir)

                # sort the image and labels to use the same pairs
                image_list.sort()
                label_list.sort()

                # iterate through the indexes
                for i in range( len(image_list) ):
                    image_path = os.path.join(image_split_path, image_list[i])
                    label_path = os.path.join(label_split_path, label_list[i])
                    executor.submit(ProcessMask, image_path, label_path, dest_image_split_dir, dest_label_split_dir, threshold=THRESHOLD)

if __name__ == "__main__":
    # -------------------------------------------------------------

    des="""
    A preprocessing utility that will use a given directory with both 
    images and label, perform center crop on the respective image and
    label, clean up the noise from the image, and binarize the label
    images to [0, 255]
    """

    # -------------------------------------------------------------
    MODALITY = ["t1c" , "t1n", "t2f" ,"t2w"] 

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--segmentation', action="store_true", help='if for segmentation, binary mask will be [0, 1, 255], not [0, 128, 255]\t[None]')
    parser.add_argument('--modality', type=str, choices=MODALITY, nargs='+', help=f'BraTS dataset modalities to use\t[t1c, t1n, t2f, t2w]')
    parser.add_argument('--crop_size', type=int, help='final NxN image crop\t[192]')
    parser.add_argument('--threshold', type=int, help='threshold for binarizing the image\t[15]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "."
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "."
    if args.crop_size is not None:
        CROP_SIZE = args.crop_size
    else: CROP_SIZE = 192
    if args.threshold is not None:
        THRESHOLD = args.threshold
    else: THRESHOLD = 10
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.modality is not None:
        MODALITY = [mod for mod in args.modality]
    SEGMENTATION = args.segmentation

    Main()
    print("\nFinish processing images, check your directory\n")