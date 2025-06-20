"""
This script binarizes the ground truth data
which is numpy file with 0-3 values, by binarizing on 
values greater than 0, thus creating a mask of the whole tumor. 
This mask data is then used to train YOLOv11-seg model. 
This script takes as an input, the ground truth directories created 
by brats_2d_slicer_YOLO, so use in conjunction. 
"""

"""Imports"""
import os
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""HYPERPARAMETERS"""
GT_FOLDER = ""
DEST_FOLDER = ""

MODALITY = ["t1c", "t1n", "t2f" ,"t2w"]  

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def BinarizeGT(input_dir, output_dir, save_as_np): 
    # input_dir is test, train, val
    gt_dir_list = os.listdir(input_dir)

    for gt_slice in gt_dir_list: 
        gt_dir = os.path.join(input_dir, gt_slice)
        
        # Load the file
        slice = np.load(gt_dir)
        
        # Binarize based on the whole tumor
        binary_mask = np.where(slice > 0, 1, 0)

        if save_as_np:
            # Save the mask in .npy
            gt_output = os.path.join(output_dir, gt_slice)
            np.save(gt_output, binary_mask)
            print(f"Saving to... {gt_output}")

        else: 
            # Get the basename without the .npy extension
            filename = os.path.basename(gt_dir).split(".")[0]

            # Save the mask as .png
            filename = f"{filename}.png"

            # Scale image back to 0-255
            binary_mask = binary_mask*255

            gt_output = os.path.join(output_dir, filename)
            cv.imwrite(gt_output, binary_mask.astype(np.uint8))

"""Main Runtime"""
def GTBinarizer(save_as_np=True): 
    for mod in MODALITY:
        gt_folder = f"{mod}/labels"
        DEST_FOLDER = f"binarized_{mod}/masks"
        # set up cwd 
        root_dir = os.getcwd()
        gt_dir = os.path.join(root_dir, gt_folder)

        # contains the test, train, val
        gt_dir_list = os.listdir(gt_dir)

        # define a thread pool executor with a maximum numnber of workers
        max_workers = 5 # adjust based on your syster's capabilities

        # Create destinations
        for dataset_split in gt_dir_list:
            CreateDir(os.path.join(DEST_FOLDER, dataset_split))

        # Use ThreadPoolExecutor 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for split in gt_dir_list:
                input_dir = os.path.join(gt_dir, split)
                output_dir = os.path.join(DEST_FOLDER, split)
                executor.submit(BinarizeGT, input_dir, output_dir, save_as_np)

        # for split in gt_dir_list:
        #     input_dir = os.path.join(gt_dir, split)
        #     output_dir = os.path.join(DEST_FOLDER, split)
        #     BinarizeGT(input_dir, output_dir, save_as_np)


if __name__ == "__main__": 
    GTBinarizer(save_as_np=False)
    print("\nFinish binarizing, please check your directory for mask\n")