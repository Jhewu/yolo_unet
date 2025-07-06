#!/bin/bash

# This function is for YOLO

python3 utils/split_dataset_YOLO.py
python3 utils/brats_2d_slicer_YOLO.py

# Insert the Crop Here And Modify the Function to Crop both the image and the mask as well
python3 utils/image_crop.py


python3 utils/binarize_gt.py
python3 utils/masks_to_boxes.py
python3 utils/copy_training_img.py

rm -r ./binarized_t1c
rm -r ./binarized_t1n
rm -r ./binarized_t2f
rm -r ./binarized_t2w

rm -r ./dataset_split

rm -r ./t1c
rm -r ./t1n
rm -r ./t2f
rm -r ./t2w