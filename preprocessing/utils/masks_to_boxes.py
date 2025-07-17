import os
import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor

def ConvertChordsToYOLO(image_width, image_height, x, y, width, height):
    """
    The code below is borrowed from Farah Alarbid
    Credits here: https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format
    """
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return x_center, y_center, normalized_width, normalized_height

def ProcessMask(mask_path):
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
            class_label = 0 
            x_center, y_center, normalized_width, normalized_height = ConvertChordsToYOLO(mask.shape[1], mask.shape[0], x, y, width, height)
            objects_info = [(class_label, x_center, y_center, normalized_width, normalized_height)]

        # class_label = 0 
        # x_center, y_center, normalized_width, normalized_height = ConvertChordsToYOLO(mask.shape[1], mask.shape[0], x, y, width, height)
        # objects_info.append((class_label, x_center, y_center, normalized_width, normalized_height))

    return objects_info

def WriteYOLOAnnotations(output_path, image_name, objects_info):
    """
    The code below is borrowed from Farah Alarbid
    Credits here: https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format
    """
    image_name = os.path.basename(image_name).split(".")[0]
    image_name = f"{image_name}.txt"
    annotation_file_path = os.path.join(output_path, image_name)
    print(f"\nThis is annotation file {annotation_file_path}")

    with open(annotation_file_path, "w") as file:
        for obj_info in objects_info:
            line = f"{obj_info[0]} {obj_info[1]} {obj_info[2]} {obj_info[3]} {obj_info[4]}\n"
            file.write(line)

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def MaskToYOLO(dir_to_examine, output_dir):
    CreateDir(output_dir)

    for mask in os.listdir(dir_to_examine):
        mask_path = os.path.join(dir_to_examine, mask)

        # Convert mask to YOLO format
        chords = ProcessMask(mask_path)
        if not chords: 
            print(f"No polygons found for {mask}")
        else: print(chords)

        WriteYOLOAnnotations(output_dir, mask, chords)
            
def Main(): 
    for mod in MODALITY:
        label_dir = f"{IN_DIR}/{mod}_segmentation/labels"
        dest_label_dir = f"{OUT_DIR}/{mod}_detection/labels"

        root = os.getcwd() 
        gt_dir = os.path.join(root, label_dir)

        # The list of splits (e.g., test, train and val)
        gt_dir_list = os.listdir(gt_dir)

        # mask to polygons on test directory 
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[0])        
            output_dir = os.path.join(dest_label_dir, gt_dir_list[0])
            executor.submit(MaskToYOLO, input_dir, output_dir)

        # mask to polygons on train directory
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[1])       
            output_dir = os.path.join(dest_label_dir, gt_dir_list[1]) 
            executor.submit(MaskToYOLO, input_dir, output_dir)

        # mask to polygons on val directory
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[2])   
            output_dir = os.path.join(dest_label_dir, gt_dir_list[2])     
            executor.submit(MaskToYOLO, input_dir, output_dir)

if __name__ == "__main__": 
    # -------------------------------------------------------------
    des="""
    An additional script that converts segmentation 
    masks (binary JPGs) into the YOLO box detection format
    """
    # -------------------------------------------------------------

    MODALITY = ["t1c" , "t1n", "t2f" ,"t2w"] 

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--modality', type=str, choices=MODALITY, nargs='+', help=f'BraTS dataset modalities to use\t[t1c, t1n, t2f, t2w]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "."
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "."
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.modality is not None:
        MODALITY = [mod for mod in args.modality]

    Main()
    print("\nFinish converting binary mask to boxes, please check your directory")