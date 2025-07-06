"""
This script is used to stack grayscale images/files 
of 3 different modality into one, .PNG file. It's part of the 
process_stack.sh workflow, do not use it alone.
"""

"""Imports"""
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2

"""HYPERPARAMETERS"""
TRAINING_FOLDERS = "_dataset"
DEST_FOLDER = "stack_dataset/images"

MODALITY = ["t1c", "t2f", "t2w"] # the modalities we are stacking

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name) 

def CombinedStack(images): 
    return np.stack(images, axis=-1) 

def ProcessImages(img_index, mod1_dir, mod0_list, mod2_dir, mod1_list, mod3_dir, mod2_list, dest_dir):
    img1_dir = os.path.join(mod1_dir, mod0_list[img_index])
    img2_dir = os.path.join(mod2_dir, mod1_list[img_index])
    img3_dir = os.path.join(mod3_dir, mod2_list[img_index])

    img1 = cv2.imread(img1_dir, cv2.IMREAD_GRAYSCALE)  # Read as grayscale if needed
    img2 = cv2.imread(img2_dir, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(img3_dir, cv2.IMREAD_GRAYSCALE)

    images = [img1, img2, img3]
    image = CombinedStack(images)

    # image_name = mod0_list[img_index].replace("-t1c.png", "")

    output_dir = os.path.join(dest_dir, mod0_list[img_index])
    print(f"Saving image to...{output_dir}")
    cv2.imwrite(f"{output_dir}", image)

"""Main Runtime"""
def Stack_2D_Slides(): 
    # set up cwd
    root_dir = os.getcwd()

    # initialize the list
    mod_to_combine = []

    # for all modality paths
    for mod in MODALITY:
        mod_to_combine.append(f"{mod}{TRAINING_FOLDERS}/images")

    split_list = ["test", "train", "val"]

    # iterate through each split
    for split in split_list:
        mod0 = os.path.join(mod_to_combine[0], split)
        mod0_list = os.listdir(mod0)

        mod1 = os.path.join(mod_to_combine[1], split)
        mod1_list = os.listdir(mod1)

        mod2 = os.path.join(mod_to_combine[2], split)
        mod2_list = os.listdir(mod2)

        # Use ThreadPoolExecutor to sort the lists in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future1 = executor.submit(sorted, mod0_list)
            future2 = executor.submit(sorted, mod1_list)
            future3 = executor.submit(sorted, mod2_list)

            # Retrieve the sorted lists
            mod0_list = future1.result()
            mod1_list = future2.result()
            mod2_list = future3.result()

        # check if they are sorted
        print(mod0_list[-5:-1])
        print(mod1_list[-5:-1])
        print(mod2_list[-5:-1])

        # create directory
        dest_dir_to_save = os.path.join(DEST_FOLDER, split)
        CreateDir(dest_dir_to_save)

        #for img_index in range(len(mod0_list)):
        #   ProcessImages(img_index, mod0, mod0_list, mod1, mod1_list, mod2, mod2_list, dest_dir_to_save)

        with ThreadPoolExecutor(max_workers=10) as executor:
             for img_index in range(len(mod0_list)):
                 executor.submit(ProcessImages, img_index, mod0, mod0_list, mod1, mod1_list, mod2, mod2_list, dest_dir_to_save)

    print("All directories copied successfully.")

if __name__ == "__main__": 
    Stack_2D_Slides()
    print("\nFinish stacking the dataset, please check your working directory")

