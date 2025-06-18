"""
Original script from computervisioneng
a guy from YouTube. I have slightly modified
it so suit my needs. 

This script takes a dir containing binary mask, and then
create polygon points in .txt file so that it can be used
as labels for YOLOv11-seg. Its input is the binary mask directories
outputted by binarize_gt.py
"""

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

INPUT_FOLDER = ""
DEST_FOLDER = ""

MODALITY = ["t1c", "t1n", "t2f" ,"t2w"]  

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def MaskToPolygons(dir_to_examine, output_dir):
    for npy_mask in os.listdir(dir_to_examine):
        npy_mask_path = os.path.join(dir_to_examine, npy_mask)

        # create output directories
        CreateDir(output_dir)

        # check if the file is a .npy file
        if not npy_mask_path.endswith('.npy'):
            print(f"Skipping non-npy file: {npy_mask_path}")
            continue

        # load the binary mask from .npy file and get its contours
        mask = np.load(npy_mask_path)
        mask = (mask * 255).astype(np.uint8)  # Ensure the mask is in the correct format
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        H, W = mask.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        # check if polygons list is empty
        if not polygons:
            print(f"No polygons found for {npy_mask}")

        # write the polygons into a text file
        output_file_path = '{}.txt'.format(os.path.join(output_dir, npy_mask)[:-4])
        print(f"Writing polygons to: {output_file_path}")
        with open(output_file_path, 'w') as f:
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {} '.format(p))
                    else:
                        f.write('{} '.format(p))
            f.close()

def Main(): 
    for mod in MODALITY:
        input_folder = f"binarized_{mod}/masks"
        DEST_FOLDER = f"{mod}_dataset/labels"

        root = os.getcwd() 
        gt_dir = os.path.join(root, input_folder)
        gt_dir_list = os.listdir(gt_dir)

        max_workers = 10 # adjust based on your syster's capabilities

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[0])        
            output_dir = os.path.join(DEST_FOLDER, gt_dir_list[0])
            executor.submit(MaskToPolygons, input_dir, output_dir)

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[1])       
            output_dir = os.path.join(DEST_FOLDER, gt_dir_list[1]) 
            executor.submit(MaskToPolygons, input_dir, output_dir)

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[2])   
            output_dir = os.path.join(DEST_FOLDER, gt_dir_list[2])     
            executor.submit(MaskToPolygons, input_dir, output_dir)

if __name__ == "__main__": 
    Main()
    print("\nFinish converting binary mask to polygon, check your directory for labels")