"""
As the name suggest, this copies the training images
to the desired directory. This is part of the process_all_mod.sh
workflow. Do not use it alone
"""

"""Imports"""
import os
from concurrent.futures import ThreadPoolExecutor
import shutil

"""HYPERPARAMETERS"""
MODALITY = ["t1c", "t1n", "t2f" ,"t2w"]

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def CopyTree(src, dst):
    # Ensure the source exists
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory '{src}' does not exist.")
    try:
        # Recursively copy the directory tree
        shutil.copytree(src, dst, dirs_exist_ok=True)  # dirs_exist_ok=True to allow copying into an existing directory
        print(f"Successfully copied {src} to {dst}")
    except Exception as e:
        print(f"Error occurred while copying: {e}")

"""Main Runtime"""
def CopyTrainingImages(): 
    for mod in MODALITY:
        # For YOLO Object Detection and Segmentation
        input_1 = f"{mod}/images"
        input_2 = f"binarized_{mod}/masks"
        dest_1 = f"{mod}_dataset/images"
        dest_2 =f"{mod}_dataset/masks"
        CreateDir(dest_1), CreateDir(dest_2)

        print(os.path.exists(input_1))
        print(os.path.exists(input_2))

        # Use ThreadPoolExecutor 
        max_workers = 5
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.submit(CopyTree, input_1, dest_1)
                executor.submit(CopyTree, input_2, dest_2)

if __name__ == "__main__": 
    CopyTrainingImages()
    print("\nFinish copying, please check your directory\n")