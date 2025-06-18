"""
This script is used to split a BraTS dataset into 
test, train and val, and it's mainly used to prepare the 
dataset for YOLO training. It takes the original training
dataset from BraTS that contains .nii.gz files, and split
it into the respective splits. The reason why we are doing
this is because if the BraTS competition ends, you have no 
way of validating your dataset or your testing set. 
"""

"""Imports"""
import os
from math import ceil
from random import shuffle, seed
import shutil
import threading

"""HYPERPARAMETERS"""
DATASET_FOLDER = "BraTS-PEDs2023_Training"
DEST_FOLDER = "dataset_split"
TRAIN_SPLIT = 0.7
VAL_TEST_SPLIT = 0.15

"""Helper Functions"""
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

"""Main Runtime"""
def Split_Dataset_YOLO(): 
    # set up cwd and test, train, val paths
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, DATASET_FOLDER)

    # list of directories in dataset (contains all patient folders)
    dataset_dir_list = os.listdir(dataset_dir)

    # report information
    dataset_length = len(dataset_dir_list)
    print(f"There is a total of: {dataset_length} patients in the directory\n")

    # creating the split through indexes
    train_index = ceil(dataset_length*TRAIN_SPLIT)
    print(f"Splitting... training set is {train_index} long")

    val_index = ceil(dataset_length*VAL_TEST_SPLIT)
    print(f"Splitting... validation set is {val_index} long")

    test_index = dataset_length-train_index-val_index
    print(f"Splitting... validation set is {test_index} long")

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
    train_dest = f"{DEST_FOLDER}/train/"
    val_dest = f"{DEST_FOLDER}/val/"
    test_dest = f"{DEST_FOLDER}/test/"

    CreateDir(train_dest)
    CreateDir(val_dest)
    CreateDir(test_dest)

    # define the threads for copying directories
    threads = []

    # Create and start thread for training data
    train_thread = threading.Thread(target=CopyFile, args=(train_list, dataset_dir, train_dest))
    threads.append(train_thread)
    train_thread.start()

    # Create and start thread for validation data
    val_thread = threading.Thread(target=CopyFile, args=(val_list, dataset_dir, val_dest))
    threads.append(val_thread)
    val_thread.start()

    # Create and start thread for test data
    test_thread = threading.Thread(target=CopyFile, args=(test_list, dataset_dir, test_dest))
    threads.append(test_thread)
    test_thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\nAll directories copied successfully.")

if __name__ == "__main__": 
    Split_Dataset_YOLO()
    print("\nFinish splitting the dataset, please check your directory\n")
