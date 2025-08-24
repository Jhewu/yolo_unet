from custom_predictor.custom_detection_predictor import CustomDetectionPredictor
import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor

import argparse
import os

import piexif
import cv2
from PIL import Image

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

def crop_with_yolo(image, shape, coords, margin_of_error):
    ### Crops with YOLO coordinates xyxy
    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    row, col = shape

    # Ensure the new coordinates stay within the image boundaries
    final_x1 = max(0,   x1 - margin_of_error)
    final_y1 = max(0,   y1 - margin_of_error)
    final_x2 = min(col, x2 + margin_of_error)
    final_y2 = min(row, y2 + margin_of_error)

    return image[int(final_y1):int(final_y2), int(final_x1):int(final_x2)], (int(final_x1), int(final_y1), int(final_x2), int(final_y2))

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image_and_metadata(pil_image, dest_path, x1, y1, x2, y2): 
    ### Save YOLO bounding box into custom metadata tag for future use
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

def crop_from_yolo(image_results, label_split_dir, image_dest_dir, label_dest_dir, verifier_dest): 
    ### Crop image using bounding boxes and create yolo_cropped/ and verifier_dataset/ datasets

    global TOTAL_PREDICTIONS

    for result in image_results: 
        boxes = result.boxes

        ### If there's a prediction... 
        if len(boxes) > 0: 
            all_coords = boxes.xyxy 

            ### If there are multiple boxes
            if len(all_coords) > 1: 

                x1 = torch.min(all_coords[:, 0]).item()
                y1 = torch.min(all_coords[:, 1]).item()
                x2 = torch.max(all_coords[:, 2]).item()
                y2 = torch.max(all_coords[:, 3]).item()

                ### MIGHT NEED THIS IN THE FUTURE
                # total_conf = np.sum(boxes.conf[:])/len(boxes.conf)

            ### If there's a single box
            else: 
                coord = boxes.xyxy[0]

                x1=int(coord[0])
                y1=int(coord[1])
                x2=int(coord[2])
                y2=int(coord[3])

                ### MIGHT NEED THIS IN THE FUTURE
                # total_conf = boxes.conf

            basename = os.path.basename(result.path)
            label_path = os.path.join(label_split_dir, basename)

            dest_image_path = os.path.join(image_dest_dir, basename)
            dest_label_path = os.path.join(label_dest_dir, basename)

            orig_img = result.orig_img
            row, col, _ = orig_img.shape
            
            cropped_image, final_coords = crop_with_yolo(orig_img, (row, col), (x1, y1, x2, y2), MARGIN_OF_ERROR)
            cropped_label, final_coords = crop_with_yolo(cv2.imread(label_path, cv2.IMREAD_UNCHANGED), (row, col),  (x1, y1, x2, y2), MARGIN_OF_ERROR)
            
            ### --------------------------------------------------------------------------------------
            ### LEAVE THIS FOR TROUBLESHOOTING

            # cropped_label = draw_square_opencv(cv2.imread(label_path), center_x, center_y, CROP_SIZE)
            # result.save(filename=dest_image_path)  # save to disk
            # cv2.imwrite(dest_label_path, cropped_label)

            ### --------------------------------------------------------------------------------------

            ### For U-Net Training (yolo_cropped dataset)
            save_image_and_metadata(Image.fromarray(cropped_image), dest_image_path, final_coords[0], final_coords[1], final_coords[2], final_coords[3])
            cv2.imwrite(dest_label_path, cropped_label)
            
            ### If there's at least 10 pixels in the cropped label, then this is a True Positive Else is False Positive
            if np.sum(cropped_label) > 10:
                cv2.imwrite(os.path.join(verifier_dest, "1", basename), cropped_image)
            else: 
                cv2.imwrite(os.path.join(verifier_dest, "0", basename), cropped_image)
            
            TOTAL_PREDICTIONS+=1
            print(f"SAVING: Prediction in... {result.path}")

        ### No prediction...
        else: 
            print(f"SKIPPING: No Prediction in... {result.path}")
            pass
    
def yolo_crop_async(): 
    """
    FUTURE ME: REORGANIZE THIS FUNCTION WITH THREADPOOLEXECUTOR, AND IMPROVE READABILITY
    """
    global TOTAL_PREDICTIONS
    image_dir = os.path.join(IN_DIR, "images")
    label_dir = os.path.join(IN_DIR, "labels")

    image_dest_dir = os.path.join(OUT_DIR, "images")
    label_dest_dir = os.path.join(OUT_DIR, "labels")

    verifier_dest_dir = "verifier_dataset"

    for split in ["test", "train", "val"]:
        image_split = os.path.join(image_dir, split)
        label_split = os.path.join(label_dir, split) 

        image_dest_split = os.path.join(image_dest_dir, split)
        label_dest_split = os.path.join(label_dest_dir, split)
        verifier_dest_split = os.path.join(verifier_dest_dir, split)

        # Construct Destination Directories
        create_dir(image_dest_split), create_dir(label_dest_split)
        create_dir(os.path.join(verifier_dest_split, "0"))
        create_dir(os.path.join(verifier_dest_split, "1"))

        image_list = os.listdir(image_split)
        image_list.sort()

        # Construct the full directories of images and labels
        image_full_paths = [os.path.join(image_split, image) for image in image_list]

        args = dict(conf=CONFIDENCE, save=False, verbose=False, device="cuda")  
        predictor = CustomDetectionPredictor(overrides=args)
        predictor.setup_model(MODEL_DIR)

        ### ------------------------------------------
        ### LEAVE FOR TROUBLE SHOOTING
        for image_path in image_full_paths:
            image_results = predictor(image_path)
            crop_from_yolo(image_results, label_split, image_dest_split, label_dest_split, verifier_dest_split)
        ### ------------------------------------------

        ### --------------------------------------------------
        ### CURRENTLY SOME ISSUES WITH THREADPOOL, DO NOT USE
        # batches = [image_full_paths[i:i + 32] for i in range(0, len(image_full_paths), 32)]
        # for batch_paths in batches:
        #     batch_results = predictor(batch_paths)
        #     with ThreadPoolExecutor(max_workers=WORKERS) as executor: 
        #         for result in batch_results:
        #             executor.submit(crop_from_yolo, result, label_dir, image_dest_dir, label_dest_dir, verifier_dest_dir)
        ### --------------------------------------------------

    print(f"\nThere were a total of {TOTAL_PREDICTIONS} predictions...")

if __name__ == "__main__": 
    # ---------------------------------------------------
    des="""
    Performs YOLO cropping on a preprocessed BraTS 2D
    dataset, to prepare them segmentation training

    Creates two directories: (1) yolo_cropped, containing
    all of the YOLO cropped images, (2) verifier_dataset
    containing the dataset to train the verifier net
    to further filter YOLO false positive detections
    """
    # ---------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument("--model_dir", type=str,help='YOLO model directory\t[None]')
    parser.add_argument("--device", type=str,help='cpu or cuda\t[cuda]')

    parser.add_argument('--confidence', type=int, help='confidence for binarizing the image\t[15]')
    parser.add_argument('--margin_of_error', type=int, help='amount of pixels to pad the crops (all sides) as a margin of error\t[30]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')

    parser.add_argument('--filter', action='store_true', help='Enable YOLO Gating, discard images under the confidence score')

    args = parser.parse_args()

    """FUTURE ME: REORGANIZE THESE"""
    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "stacked_segmentation"
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "yolo_cropped"
    if args.model_dir is not None:
        MODEL_DIR = args.model_dir
    else: MODEL_DIR = "yolo_weights/best.pt"
    if args.device is not None:
        DEVICE = args.device
    else: DEVICE = "cuda"
    if args.confidence is not None:
        CONFIDENCE = args.confidence
    else: CONFIDENCE = 0.4
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.filter is not None:
        FILTER = args.filter
    else: FILTER = False
    if args.margin_of_error is not None:
        MARGIN_OF_ERROR = args.margin_of_error
    else: MARGIN_OF_ERROR = 30

    TOTAL_PREDICTIONS = 0

    yolo_crop_async()
