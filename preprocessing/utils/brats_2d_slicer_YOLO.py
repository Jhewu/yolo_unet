import os
import nibabel as nib
import cv2 as cv
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor

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

# -----------------------------------------------------------------------------------------------
# Main Runtime
# -----------------------------------------------------------------------------------------------
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
    # -------------------------------------------------------------

    des="""
    This script uses as an input .nib files and then "slices" them into 2D slices
    alongside the axial view from the range MIN_SLICE and MAX_SLICE. The input it's
    a "dataset_split/train/nib1..." directory obtained after running split_dataset_YOLO.py.
    """

    # -------------------------------------------------------------

    """HYPERPARAMETERS"""
    DATASET_FOLDER = "dataset_split"
    DEST_FOLDER = ""
    DEST_FOLDER_GT = ""

    MIN_SLICE = 30
    MAX_SLICE = 120

    MODALITY = ["t1c" , "t1n", "t2f" ,"t2w"] 

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--train_split', type=float, default=0.7, help='train split percentage\t[0.7]')
    """CHANGE THIS TO BE MORE USER FRIENDLY"""
    parser.add_argument('--val_test_split', type=float, default=0.50, help='test and validation split percentage\t[0.15]')
    args = parser.parse_args()

    if args.in_dir is not None:
        in_dir = args.in_dir
    else: raise IOError
    if args.out_dir is not None:
        out_dir = args.out_dir
    else: out_dir = "dataset_split"
    if args.train_split is not None:
        train_split = args.train_split
    else: train_split = 0.70
    if args.val_test_split is not None:
        val_test_split = args.val_test_split
    else: val_test_split = 0.50
    

    print(f"Creating slices from {MIN_SLICE} to {MAX_SLICE} (representing the z-coordinates) in axial view...\n")
    BraTS_2D_Slicer_YOLO()
    print("\nFinish slicing, please check your directory\n")

