from concurrent.futures import ThreadPoolExecutor
from custom_predictor.custom_detection_predictor import CustomDetectionPredictor
from ultralytics import YOLO
import numpy as np
import piexif
import argparse
import os
import cv2
from PIL import Image

def convert_to_xyxy(shape, coords, margin_of_error=0):
    # Parse input as (center_x, center_y, width, height)
    cx, cy, w, h = coords
    row, col, _ = shape  # Assuming shape is (height, width)

    # Convert from center-based to top-left and bottom-right coordinates
    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2

    # Expand the box by margin_of_error on each side
    final_x1 = int(max(0, x1 - margin_of_error))
    final_y1 = int(max(0, y1 - margin_of_error))
    final_x2 = int(min(col, x2 + margin_of_error))
    final_y2 = int(min(row, y2 + margin_of_error))

    return ()

def crop_with_yolo(image, shape, coords, margin_of_error=0):


    # Return the cropped image
    return image[final_y1:final_y2, final_x1:final_x2]
def crop_center(image, x_center, y_center, crop_size):
    """
    Crops an image around a center point, padding with zeros if necessary.
    
    Args:
        image: Input image (numpy array)
        x_center, y_center: Center coordinates for the crop
        crop_size: Size of the output crop (assumes square crop)
    
    Returns:
        Cropped image of size (crop_size, crop_size)
    """
    height, width = image.shape[0], image.shape[1]
    half_crop = crop_size // 2
    
    # Calculate desired crop boundaries
    x1 = x_center - half_crop
    y1 = y_center - half_crop
    x2 = x1 + crop_size  # Ensure exact crop_size
    y2 = y1 + crop_size
    
    # Calculate actual crop boundaries (clipped to image)
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(width, x2)
    y2_clip = min(height, y2)
    
    # Extract the portion of image within bounds
    cropped = image[y1_clip:y2_clip, x1_clip:x2_clip]
    
    # Calculate padding needed
    pad_left = x1_clip - x1
    pad_top = y1_clip - y1  
    pad_right = x2 - x2_clip
    pad_bottom = y2 - y2_clip
    
    # Apply padding if necessary
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        if len(image.shape) == 3:  # Color image
            cropped = np.pad(cropped, 
                           ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                           mode='constant', constant_values=0)
        else:  # Grayscale image
            cropped = np.pad(cropped, 
                           ((pad_top, pad_bottom), (pad_left, pad_right)), 
                           mode='constant', constant_values=0)
    
    return cropped

def draw_square_opencv(image, x_center, y_center, square_size, thickness=1, color=255):
    ### --------------------------------------------------------------------------------------
    ### LEAVE THIS FOR TROUBLESHOOTING
    ### --------------------------------------------------------------------------------------
    result_image = image.copy()
    half_size = square_size // 2
    
    pt1 = (x_center - half_size, y_center - half_size)
    pt2 = (x_center + half_size, y_center + half_size)
    
    if len(image.shape) == 3:
        if isinstance(color, (int, float)):
            draw_color = (color, color, color)
        else:
            draw_color = color
    else:
        draw_color = color
    
    # Use -1 for filled rectangle, positive value for border thickness
    cv_thickness = -1 if thickness >= square_size // 2 else thickness
    
    cv2.rectangle(result_image, pt1, pt2, draw_color, cv_thickness)
    return result_image

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_yolo_box(image, center_x, center_y, width, height, color_val=255):
    img_h, img_w = image.shape[:2]

    # Calculate top-left and bottom-right corner coordinates
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    # Ensure coordinates are within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # Modify pixels within the square to the specified color_val
    image[y1:y2, x1:x2] = color_val

    return image

def save_image_and_metadata(pil_image, dest_path, x1, y1, x2, y2): 
    exif = pil_image.getexif()
    exif_bytes = exif.tobytes()
    exif_dict = piexif.load(exif_bytes)

    # Convert the custom metadata to a format that can be written in EXIF
    new_data = {
        piexif.ExifIFD.UserComment: f"{x1},{y1},{x2},{y2}".encode('utf-8')
    }
    
    exif_dict["Exif"].update(new_data)
    
    # Create the bytes for writing to the image
    exif_bytes = piexif.dump(exif_dict)
    
    # Save the image with the new EXIF data
    pil_image.save(dest_path, exif=exif_bytes)

def crop_from_gt(image_path, label_path, coords, image_dest_dir, label_dest_dir): 
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    dest_image_path = os.path.join(image_dest_dir, os.path.basename(image_path))
    dest_label_path = os.path.join(label_dest_dir, os.path.basename(label_path))

    if len(coords) > 1:
        cropped_image = crop_with_yolo(image, image.shape, coords)
        cropped_label = crop_with_yolo(label, image.shape, coords)
        
        cv2.imwrite(dest_image_path, cropped_image)
        cv2.imwrite(dest_label_path, cropped_label)
    # else: 
    #     cv2.imwrite(dest_image_path, image)
    #     cv2.imwrite(dest_label_path, label)

def GetMaskCoordinates(mask_path):
    """
    The code below is borrowed from Farah Alarbid
    Credits here: https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # image processing
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects_info = []

    max_area = 0
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if max_area < (width*height):
            max_area = (width*height)
            x_center = (x + width / 2) 
            y_center = (y + height / 2)
            objects_info = (x_center, y_center, width, height)

    return objects_info

def ground_truth_crop(): 
    """
    COMMENT: REORGANIZE THIS FUNCTION WITH THREADPOOLEXECUTOR AND ONLY RUN ON CPU MODE
    """
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
        image_list.sort()
        label_list.sort()

        # Construct the full directories of images and labels
        image_full_paths = [os.path.join(image_split_dir, image) for image in image_list]
        label_full_paths = [os.path.join(label_split_dir, label) for label in label_list]

        create_dir(image_split_dest_dir), create_dir(label_split_dest_dir)

        ### ------------------------------------------
        ### LEAVE FOR TROUBLE SHOOTING
        for i, label_path in enumerate(label_full_paths):
            coords = GetMaskCoordinates(label_path)
            crop_from_gt(image_full_paths[i], label_path, coords, image_split_dest_dir, label_split_dest_dir)
        ### ------------------------------------------

        # with ThreadPoolExecutor(max_workers=WORKERS) as executor: 
        #     for image_path in image_full_paths:
        #         image_results = predictor(image_path)
        #         executor.submit(crop_from_yolo, image_results, label_split_dir, image_split_dest_dir, label_split_dest_dir)

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
    parser.add_argument("--device", type=str,help='cpu or cuda\t[cuda]')

    parser.add_argument('--confidence', type=int, help='confidence for binarizing the image\t[15]')
    parser.add_argument('--crop_size', type=int, help='final NxN image crop\t[64]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    parser.add_argument('--batch_size', type=int, help='batch size used to process YOLO, depending on your GPU capabilities\t[64]')

    parser.add_argument('--filter', action='store_true', help='Enable YOLO Gating, discard images under the confidence score')

    args = parser.parse_args()

    """REORGANIZE THESE IN THE FUTURE"""
    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "stacked_segmentation"
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "ground_truth_cropped"
    if args.model_dir is not None:
        MODEL_DIR = args.model_dir
    else: MODEL_DIR = "yolo_weights/best.pt"
    if args.device is not None:
        DEVICE = args.device
    else: DEVICE = "cuda"

    if args.crop_size is not None:
        CROP_SIZE = args.crop_size
    else: CROP_SIZE = 128
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    else: BATCH_SIZE = 256
    if args.confidence is not None:
        CONFIDENCE = args.confidence
    else: CONFIDENCE = 0.8
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.filter is not None:
        FILTER = args.filter
    else: FILTER = False

    STACK_PREDICTION = False
    TOTAL_PREDICTIONS = 0
    MARGIN_OF_ERROR = 10

    ground_truth_crop()
