"""
An additional script that converts segmentation 
masks into the YOLO format
"""

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ultralytics.utils.ops import segments2boxes

import pandas as pd
from IPython.display import display
from PIL import Image, ImageDraw

INPUT_FOLDER = ""
DEST_FOLDER = ""

MODALITY = ["t1c", "t1n", "t2f" ,"t2w"]  

"""
The code below is borrowed from Farah Alarbid
Credits here: https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format
"""
def ProcessMask(mask_path):
    # Read the Mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Image processing
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects_info = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        class_label = 0 
        x_center, y_center, normalized_width, normalized_height = ConvertChordsToYOLO(mask.shape[1], mask.shape[0], x, y, width, height)
        objects_info.append((class_label, x_center, y_center, normalized_width, normalized_height))

    return objects_info

def ConvertChordsToYOLO(image_width, image_height, x, y, width, height):
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return x_center, y_center, normalized_width, normalized_height

def write_yolo_annotations(output_path, image_name, objects_info):
    image_name = os.path.basename(image_name).split(".")[0]
    image_name = f"{image_name}.txt"
    annotation_file_path = os.path.join(output_path, image_name)
    print(f"\nThis is annotation file {annotation_file_path}")

    with open(annotation_file_path, "w") as file:
        for obj_info in objects_info:
            line = f"{obj_info[0]} {obj_info[1]} {obj_info[2]} {obj_info[3]} {obj_info[4]}\n"
            file.write(line)

"""------"""


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

        write_yolo_annotations(output_dir, mask, chords)


        

        # # Write the Boxes into a txt file
        # output_file_path = '{}.txt'.format(os.path.join(output_dir, mask)[:-4])
        # print(f"Writing polygons to: {output_file_path}")
        # with open(output_file_path, 'w') as f:
        #     for polygon in polygons:
        #         for p_, p in enumerate(polygon):
        #             if p_ == len(polygon) - 1:
        #                 f.write('{}\n'.format(p))
        #             elif p_ == 0:
        #                 f.write('0 {} '.format(p))
        #             else:
        #                 f.write('{} '.format(p))
        #     f.close()





        # # load the binary mask from .npy file and get its contours
        # mask = np.load(mask_path)
        # mask = (mask * 255).astype(np.uint8)  # Ensure the mask is in the correct format
        # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # H, W = mask.shape
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # convert the contours to polygons
        # polygons = []
        # for cnt in contours:
        #     if cv2.contourArea(cnt) > 200:
        #         polygon = []
        #         for point in cnt:
        #             x, y = point[0]
        #             polygon.append(x / W)
        #             polygon.append(y / H)
        #         polygons.append(polygon)

        # # check if polygons list is empty
        # if not polygons:
        #     print(f"No polygons found for {npy_mask}")

        # # write the polygons into a text file
        # output_file_path = '{}.txt'.format(os.path.join(output_dir, npy_mask)[:-4])
        # print(f"Writing polygons to: {output_file_path}")
        # with open(output_file_path, 'w') as f:
        #     for polygon in polygons:
        #         for p_, p in enumerate(polygon):
        #             if p_ == len(polygon) - 1:
        #                 f.write('{}\n'.format(p))
        #             elif p_ == 0:
        #                 f.write('0 {} '.format(p))
        #             else:
        #                 f.write('{} '.format(p))
        #     f.close()






    # for polygons in os.listdir(dir_to_examine): 
    #     coords = []

    #     polygon_path = os.path.join(dir_to_examine, polygons)
    #     with open(polygon_path, "r") as f: 
    #         segments = f.read().split()
    #         if not segments: 
    #             print("empty")
    #             continue
    #         else: 
    #             print("not empty")
    #             segments.remove("0")
    #             segments = list(map(float, segments))
    #             print(segments)
    #             new_segments = np.array([segments])
    #             new_choords = segments2boxes([s.reshape(-1, 2) for s in new_segments])
    #             print(f"\n{new_choords}")
    #             print()



            
    
def Main(): 
    for mod in MODALITY:
        input_folder = f"binarized_{mod}/masks"
        DEST_FOLDER = f"{mod}_dataset/labels"

        root = os.getcwd() 
        gt_dir = os.path.join(root, input_folder)

        # The list of splits (e.g., test, train and val)
        gt_dir_list = os.listdir(gt_dir)

        max_workers = 5 # adjust based on your syster's capabilities

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[0])        
            output_dir = os.path.join(DEST_FOLDER, gt_dir_list[0])
        # MaskToYOLO(input_dir, output_dir)
            executor.submit(MaskToYOLO, input_dir, output_dir)

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[1])       
            output_dir = os.path.join(DEST_FOLDER, gt_dir_list[1]) 
            executor.submit(MaskToYOLO
        , input_dir, output_dir)

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[2])   
            output_dir = os.path.join(DEST_FOLDER, gt_dir_list[2])     
            executor.submit(MaskToYOLO
        , input_dir, output_dir)

if __name__ == "__main__": 
    Main()
    print("\nFinish converting binary mask to polygon, check your directory for labels")