from concurrent.futures import ThreadPoolExecutor
from custom_predictor.custom_detection_predictor import CustomDetectionPredictor
from ultralytics import YOLO
import numpy as np
import argparse
import os
import cv2

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

def crop_from_yolo(image_results, label_split_dir, image_dest_dir, label_dest_dir): 
    global TOTAL_PREDICTIONS
    # Check if there's any objects with a prediction, and obtain such coordinates
    for result in image_results: 
        boxes = result.boxes
        if len(boxes) > 0: 
            all_coords = boxes.xywh
            
            if len(all_coords) > 1: 
                total_x, total_y = 0, 0

                for coord in all_coords:
                    total_x+=coord[0]
                    total_y+=coord[1]

                # Obtain the "centroid" of all_chords
                center_x = int(total_x/len(all_coords))
                center_y = int(total_y/len(all_coords))                    
            else: 
                coords = boxes.xywh[0]
                center_x = int(coords[0])
                center_y = int(coords[1])

            orig_img = result.orig_img
            basename = os.path.basename(result.path)
            label_path = os.path.join(label_split_dir, basename)

            cropped_image = crop_center(orig_img, center_x, center_y, CROP_SIZE)
            cropped_label = crop_center(cv2.imread(label_path, cv2.IMREAD_UNCHANGED), center_x, center_y, CROP_SIZE)

            dest_image_path = os.path.join(image_dest_dir, basename)
            dest_label_path = os.path.join(label_dest_dir, basename)

            ### --------------------------------------------------------------------------------------
            ### LEAVE THIS FOR TROUBLESHOOTING

            # cropped_label = draw_square_opencv(cv2.imread(label_path), center_x, center_y, CROP_SIZE)
            # result.save(filename=dest_image_path)  # save to disk
            # cv2.imwrite(dest_label_path, cropped_label)

            ### --------------------------------------------------------------------------------------

            cv2.imwrite(dest_image_path, cropped_image)
            cv2.imwrite(dest_label_path, cropped_label)

            TOTAL_PREDICTIONS+=1
            print(f"SAVING: Prediction in... {result.path}")
        else: 
            print(f"SKIPPING: No Prediction in... {result.path}")
            pass

def yolo_crop_async(): 
    """
    COMMENT: REORGANIZE THIS FUNCTION WITH THREADPOOLEXECUTOR AND ONLY RUN ON CPU MODE
    """

    global TOTAL_PREDICTIONS
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

        # Ensure image matches label
        image_list.sort()

        # Construct the full directories of images and labels
        image_full_paths = [os.path.join(image_split_dir, image) for image in image_list]
        create_dir(image_split_dest_dir), create_dir(label_split_dest_dir)

        args = dict(conf=CONFIDENCE, save=False, verbose=False)  
        predictor = CustomDetectionPredictor(overrides=args)
        predictor.setup_model(MODEL_DIR)

        # Batch the directories
        for i in range(0, len(image_full_paths)): 
            
            # Create the batches
            image_paths = image_full_paths[i]

            # Load Custom Predictor and crop each image
            image_results = predictor(image_paths)

            ### ----------------------------------------------------------------------------------------
            ### Save This For Troubleshooting

            crop_from_yolo(image_results, label_split_dir, image_split_dest_dir, label_split_dest_dir)
            ### ----------------------------------------------------------------------------------------

            # Crop the images from YOLO coordinates (labels use the same crop coordinates as image)
            # with ThreadPoolExecutor(max_workers=WORKERS) as executor: 
                # executor.submit(crop_from_yolo, image_results, label_split_dir, image_split_dest_dir, label_split_dest_dir)

    print(f"\nThere were a total of {TOTAL_PREDICTIONS} predictions...")

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
    else: CROP_SIZE = 128
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    else: BATCH_SIZE = 256
    if args.confidence is not None:
        CONFIDENCE = args.confidence
    else: CONFIDENCE = 0.6
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.filter is not None:
        FILTER = args.filter
    else: FILTER = False
    if args.use_cpu is not None:
        USE_CPU = args.use_cpu
    else: USE_CPU = False

    TOTAL_PREDICTIONS = 0

    yolo_crop_async()
