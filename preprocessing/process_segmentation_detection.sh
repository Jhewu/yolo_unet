#!/bin/bash

## -------------------------------------------------
## This script it's for processing BraTS dataset for
## both YOLO object detection, and U-NET Training

## After running this bash script, it will create
## temporary directions and remove them, leaving three
## directories: (1) stacked_detection, and (2) stacked_segmentation
## for each respective tasks (all four modalities stacked onto 
## each PNG channel)
## -------------------------------------------------

## For UNET Segmentation
python3 utils/split_dataset.py --in_dir  BraTS2025-GLI-PRE-Challenge-TrainingData # ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2 
python3 utils/brats_2d_slicer.py
python3 utils/crop_clean_binarize.py 

## For YOLO Object Detection
python3 utils/masks_to_boxes.py 
python3 utils/copy_training_img.py --dataset_to_copy_from segmentation --dataset_to_copy_to detection

## For YOLO Segmentation
python3 utils/masks_to_polygons.py
python3 utils/copy_training_img.py --dataset_to_copy_from segmentation --dataset_to_copy_to yoloseg

## For Stacking Images
python3 utils/stack_images.py --dataset segmentation
python3 utils/copy_labels.py --dataset segmentation

python3 utils/stack_images.py --dataset detection
python3 utils/copy_labels.py --dataset detection

## Remove Trash (Optional)
rm -r ./dataset_split
rm -r ./t1c
rm -r ./t1n
rm -r ./t2f
rm -r ./t2w

# Remove Single Modality Dataset
rm -r ./t1c_segmentation
rm -r ./t1n_segmentation
rm -r ./t2f_segmentation
rm -r ./t2w_segmentation

rm -r ./t1c_detection
rm -r ./t1n_detection
rm -r ./t2f_detection
rm -r ./t2w_detection
