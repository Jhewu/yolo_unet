import os
import cv2
import numpy as np
from shutil import copy2

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_dataset(): 
    image_dir = os.path.join(IN, "images")
    label_dir = os.path.join(IN, "labels")

    for split in SPLIT: 
        image_split = os.path.join(image_dir, split)
        label_split = os.path.join(label_dir, split)

        dest_split = os.path.join(OUT, split)

        zero = os.path.join(dest_split, "0")
        one = os.path.join(dest_split, "1")        
        create_dir(zero), create_dir(one)

        labels = os.listdir(label_split)

        for label in labels:
            label_img = cv2.imread(os.path.join(label_split, label), cv2.IMREAD_GRAYSCALE)
            # print(label_img)
            filename = os.path.basename(label)

            if np.sum(label_img) > 5: 
                copy2(os.path.join(image_split, filename), os.path.join(one, filename))
            else: 
                copy2(os.path.join(image_split, filename), os.path.join(zero, filename))
                
        # print(os.path.exists(image_split))

    

    #     ### If there's at least 10 pixels in the cropped label, then this is a True Positive Else is False Positive
    # if np.sum(cropped_label) > 10:
    #     cv2.imwrite(os.path.join(verifier_dest, "1", basename), cropped_image)
    #     # pass
    # else: 
    #     cv2.imwrite(os.path.join(verifier_dest, "0", basename), cropped_image)
    #     # pass
    

if __name__ == "__main__": 
    IN = "ssa_stacked_segmentation"
    OUT = "ssa_classification_dataset"
    SPLIT = ["test", "train", "val"]

    create_dataset()