import os
import cv2 as cv
import shutil

"""
Creates new dataset directory with all of the empty masks 
removed

TODO: Incorporate thread pool executor, to parallelize processing
"""

DATASET_DIR = "t1c"
DEST_DIR = f"cleaned_{DATASET_DIR}"

def isAllZero(image):
    return cv.countNonZero(image) == 0

def copyFile(src, dst): 
    shutil.copy(src, dst)

def createDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def dataCleaner():
    data_subdir = os.listdir(DATASET_DIR)

    # mask dir
    mask_dir = os.path.join(DATASET_DIR, data_subdir[1])
    mask_dir_des = os.path.join(DEST_DIR, data_subdir[1])
    mask_subdir = os.listdir(mask_dir)

    # image dir
    image_dir = os.path.join(DATASET_DIR, data_subdir[0])
    image_dir_des = os.path.join(DEST_DIR, data_subdir[0])

    # iterate through test, train, val
    for split in mask_subdir[:]: 
        mask_split = os.path.join(mask_dir, split)
        image_split = os.path.join(image_dir, split)

        # create destination directories
        mask_split_des = os.path.join(mask_dir_des, split)
        image_split_des = os.path.join(image_dir_des, split)
        createDir(mask_split_des)
        createDir(image_split_des)

        # list containing the individual images/masks names
        # could also get rid of image_list, since they are equivalent
        mask_lists = os.listdir(mask_split)

        # iterate through the masks
        for mask in mask_lists[:]: 
            
            # read the image
            mask_path = os.path.join(mask_split, mask)
            image_path = os.path.join(image_split, mask)
            mask_image = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

            # check if it's all zero
            if isAllZero(mask_image):
                print(f"Removed: {mask}")
            else:
                # if it's not zero, then copy to the new destination
                mask_des_path = os.path.join(mask_split_des, mask)
                image_des_path = os.path.join(image_split_des, mask)
                copyFile(mask_path, mask_des_path)
                copyFile(image_path, image_des_path)

                print(f"Copied {mask} to: {mask_des_path}")
                print(f"Copied {mask} to: {image_des_path}")

if __name__ == "__main__": 
    dataCleaner()