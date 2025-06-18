"""
Since my ground truth mask are .npy masks
and mask_to_polygons.py requires .png, this
script ensure I can still use mask_to_polygons.py
"""

import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor

FOLDER_NAME = ""
DEST_DIR = ""

MODALITY = ["t1c", "t1n", "t2f" ,"t2w"]  

"""Helper Function"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def ConvertNpyToPng(npy_path, output_dir):
    # Load the .npy file
    image_array = np.load(npy_path)

    # Normalize the image array to 0-255
    image_array = (image_array * 255).astype(np.uint8)

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the output file path
    base_filename = os.path.basename(npy_path).replace('.npy', '.png')
    output_path = os.path.join(output_dir, base_filename)

    # Save the image as a .jpg file
    cv2.imwrite(output_path, image_array)
    print(f"Saved {output_path}")

"""Main Runtime"""
def Main(): 
    for mod in MODALITY:
        gt_folder = f"binarized_masks/binarized_{mod}/masks"
        DEST_FOLDER = f"final_dataset/{mod}/masks"
        # set up cwd 
        root_dir = os.getcwd()
        gt_dir = os.path.join(root_dir, gt_folder)

        # contains the test, train, val
        gt_dir_list = os.listdir(gt_dir)

        # define a thread pool executor with a maximum numnber of workers
        max_workers = 10 # adjust based on your syster's capabilities

        # Create destinations
        for dataset_split in gt_dir_list:
            CreateDir(os.path.join(DEST_FOLDER, dataset_split))

        for split in gt_dir_list: 
            input_dir = os.path.join(gt_dir, split)
            output_dir = os.path.join(DEST_FOLDER, split)            
            input_dir_list = os.listdir(input_dir)
            
            # Use ThreadPoolExecutor 
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for image in input_dir_list: 
                    img_path = os.path.join(input_dir, image)
                    executor.submit(ConvertNpyToPng, img_path, output_dir)

if __name__ == "__main__": 
    Main()
    print(f"\nFinish converting .npy to jpg, check your directory for {DEST_DIR}")