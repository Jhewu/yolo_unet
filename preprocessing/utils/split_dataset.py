import os
import argparse
from math import ceil
from random import shuffle, seed
import shutil
import threading

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name) 
       
def CopyFile(dataset_dir_list, dataset_dir, dataset_dest):
    # dir is the patient directory
    for dir in dataset_dir_list:
        dir_to_copy = os.path.join(dataset_dir, dir)
        dir_to_copy_to = os.path.join(dataset_dest, dir)
        if os.path.exists(dir_to_copy):
            shutil.copytree(dir_to_copy, dir_to_copy_to)
        else:
            print(f"Source directory does not exist: {dir_to_copy}")

# ------------------------------------------------------------------------------
# Main Runtime
# ------------------------------------------------------------------------------
def Split_Dataset_YOLO(): 
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, IN_DIR)

    # list of directories in dataset (contains all patient folders)
    dataset_dir_list = os.listdir(dataset_dir)

    # report information
    dataset_length = len(dataset_dir_list)
    print(f"There is a total of: {dataset_length} patients in the directory\n")

    # creating the split through indexes
    train_index = ceil(dataset_length*TRAIN_SPLIT)
    print(f"Splitting... training set is {train_index} long")

    # first percetange it's for validation, second for test
    val_split = abs(VAL_TEST_SPLIT)
    test_split = abs(1 - VAL_TEST_SPLIT)

    val_index = ceil( 
        (abs(dataset_length-(dataset_length*TRAIN_SPLIT)))* val_split)
    print(f"Splitting... validation set is {val_index} long")

    test_index = ceil( 
        (abs(dataset_length-(dataset_length*TRAIN_SPLIT)))* test_split)
    print(f"Splitting... test set is {test_index} long")

    # randomly shuffle the list before splitting 
    # use a seed to shuffle indices to ensure 
    # the same order on both list
    seed(42)
    shuffle(dataset_dir_list)

    # create the respective train, val and test split
    train_list = dataset_dir_list[:train_index]
    val_list = dataset_dir_list[train_index:train_index+val_index]
    test_list = dataset_dir_list[train_index+val_index:]

    # create directories for each split
    train_dest = f"{OUT_DIR}/train/"
    val_dest = f"{OUT_DIR}/val/"
    test_dest = f"{OUT_DIR}/test/"

    CreateDir(train_dest), CreateDir(val_dest), CreateDir(test_dest)

    # define the threads for copying directories
    threads = []

    train_thread = threading.Thread(target=CopyFile, args=(train_list, dataset_dir, train_dest))
    threads.append(train_thread)
    train_thread.start()

    val_thread = threading.Thread(target=CopyFile, args=(val_list, dataset_dir, val_dest))
    threads.append(val_thread)
    val_thread.start()

    test_thread = threading.Thread(target=CopyFile, args=(test_list, dataset_dir, test_dest))
    threads.append(test_thread)
    test_thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\nAll directories copied successfully.")

if __name__ == "__main__": 
    # -------------------------------------------------------------

    des="""
    This script split a BraTS dataset into test, train and val.
    It uses the original training BraTS dataset containing
    .nii.gz files as input. The reason why we are doing that it's 
    because once the BraTS competition ends, you have no 
    way of validating your dataset or your testing set. 
    """

    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--train_split', type=float, default=0.7, help='train split percentage\t[0.7]')
    parser.add_argument('--val_test_split', type=float, default=0.50, help='validation and test split percentage after train split\t[0.50]')
    args = parser.parse_args()

    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: raise IOError
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "dataset_split"
    if args.train_split is not None:
        TRAIN_SPLIT = args.train_split
    else: TRAIN_SPLIT = 0.70
    if args.val_test_split is not None:
        VAL_TEST_SPLIT = args.val_test_split
    else: VAL_TEST_SPLIT = 0.50
    
    Split_Dataset_YOLO()
    print("\nFinish splitting the dataset, please check your directory\n")
