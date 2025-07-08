#!/bin/bash

# -------------------------------------------------
# This script it's for processing BraTS dataset for
# both YOLO object detection and UNET Training

# -> {mod}_detection directories are for YOLO detection
# -> {mod}_segmentation directories are for UNET
# -------------------------------------------------

# For UNET
python3 utils/split_dataset.py --in_dir ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2 
python3 utils/brats_2d_slicer.py
python3 utils/crop_clean_binarize.py
# For YOLO
python3 utils/masks_to_boxes.py
python3 utils/copy_training_img.py

# Remove trash
rm -r ./dataset_split
rm -r ./t1c
rm -r ./t1n
rm -r ./t2f
rm -r ./t2w