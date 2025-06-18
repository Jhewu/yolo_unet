"""
This script is specifically made to prepare data for Yolov11-seg 
training. It takes in as an input a  "dataset/split" directory (with
test, val, train directories) that you obtain after running process
split_dataset_YOLO.py. From there, it creates 2D slices from the range 
MIN_SLICE and MAX_SLICE (z-coordinates)in axial view of the brain. By 
default, it creates one modality, but you can use process_all_mod.sh 
to run all. By default it saves the 2d slices as .npy files 
so that there's no need to normalize. You can modify that
by changing the save_as_np to False in the function GetImageSlices()
Only the training 2d slices are normalized, the ground_truth 
is kept the default 0-3 range for easier mask extraction during training. 
If you still want to normalize it, you can disable it by turning is_ground_true = False
"""

"""Imports"""
import os
import nibabel as nib
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

"""HYPERPARAMETERS"""
DATASET_FOLDER = "dataset_split"
DEST_FOLDER = ""
DEST_FOLDER_GT = ""

MIN_SLICE = 30
MAX_SLICE = 120

MODALITY = ["t1c" , "t1n", "t2f" ,"t2w"] 

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def GetImageSlices(modality_name, file_dir, file_dir_dest, mod, save_as_np=True, is_ground_truth=False): 
    print(f"Slicing at...{file_dir}")

    # convert raw data
    file = nib.load(file_dir)
    file = file.get_fdata()
    
    # save slices from MIN_SLICE and MAX_SLICE
    for slice in range(MIN_SLICE, MAX_SLICE):
        # get 2d slice 
        slice_2d = file[:, :, slice]

        # normalize the 2d slice to the range of [0, 1]
        slice_2d_min = np.min(slice_2d) 
        slice_2d_max = np.max(slice_2d)
        
        if is_ground_truth == False:
        # for ground truth, the default range is [0-3]. We
        # will not normalize it because then we can easily 
        # extract the ground truth mask, if you still want to 
        # normalize it, you can disable it by turning is_ground_true = False
            if slice_2d_max > 0:
                slice_2d = (slice_2d - slice_2d_min) / slice_2d_max 
            else: 
                print(f"No value in this ground truth.") 
                slice_2d = slice_2d - slice_2d_min # At least shift the min to 0
        
        # save the slice 2d into the respective directory
        slice_name = os.path.join(file_dir_dest, f"{modality_name}{slice}-{mod}")
        if save_as_np: 
            np.save(slice_name, slice_2d)
        else: 
            slice_2d = (slice_2d * 255).astype(np.uint8)
            cv.imwrite(f"{slice_name}.png", slice_2d)

def GetPatientScan(root_dir, dest_dir, dest_dir_gt, mod): 
    # list of directories in each patient folder
    patient_list = os.listdir(root_dir)

    print(dest_dir)

    for patient in patient_list: 
        patient_dir = os.path.join(root_dir, patient)
        patient_dir_list = os.listdir(patient_dir)
        for file in patient_dir_list: 
            
            ground_truth = f"{patient}-seg.nii.gz"
            chosen_modality = f"{patient}-{mod}.nii.gz"
            file_dir = os.path.join(patient_dir, file)

            # create the destination directories
            modality_name = file.replace(".nii.gz", "")

            # separate the ground truth
            if file == ground_truth: 
                modality_name = modality_name.replace("-seg", "")
                GetImageSlices(modality_name, file_dir, dest_dir_gt, mod, save_as_np=True, is_ground_truth=True)
            elif file == chosen_modality:
                modality_name = modality_name.replace(f"-{mod}", "")
                GetImageSlices(modality_name, file_dir, dest_dir, mod, save_as_np=False, is_ground_truth=False)

"""Main Runtime"""
def BraTS_2D_Slicer_YOLO(): 
    for mod in MODALITY:
        # change all folder hyperparameters
        DATASET_FOLDER = "dataset_split"
        DEST_FOLDER = f"{mod}/images"
        DEST_FOLDER_GT = f"{mod}/labels"

        # set up cwd and dataset split dir
        root_dir = os.getcwd()
        dataset_dir = os.path.join(root_dir, DATASET_FOLDER)

        # list of the splits (test, train, val)
        dataset_dir_list = os.listdir(dataset_dir)

        # define a thread pool executor with a maximum numnber of workers
        max_workers = 10 # adjust based on your syster's capabilities

        # Ensure destination directories exist
        for dataset_split in dataset_dir_list:
            CreateDir(os.path.join(DEST_FOLDER, dataset_split))
            CreateDir(os.path.join(DEST_FOLDER_GT, dataset_split))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for split in dataset_dir_list:
                input_dir = os.path.join(dataset_dir, split)
                output_dir = os.path.join(DEST_FOLDER, split)
                output_dir_gt = os.path.join(DEST_FOLDER_GT, split)
                executor.submit(GetPatientScan, input_dir, output_dir, output_dir_gt, mod)

        print("All directories processed successfully.")

if __name__ == "__main__": 
    print(f"Creating slices from {MIN_SLICE} to {MAX_SLICE} (representing the z-coordinates) in axial view...\n")
    BraTS_2D_Slicer_YOLO()
    print("\nFinish slicing, please check your directory\n")

