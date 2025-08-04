#!/bin/bash

## -------------------------------------------------
## This script it's for processing BraTS dataset for
## YOLO Segmentation ONLY (in Ultralytics format)

## After running this bash script, it will create
## temporary directions and remove them, leaving one
## directory "yolo_seg" (all four modalities stacked onto 
## each PNG channel)
## -------------------------------------------------

## Preprocessing
python3 utils/split_dataset.py --in_dir BraTS2025-GLI-PRE-Challenge-TrainingData # ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2 
python3 utils/brats_2d_slicer.py
python3 utils/crop_clean_binarize.py --segmentation

## For YOLO Segmentation
python3 utils/masks_to_polygons.py
python3 utils/copy_training_img.py --dataset_to_copy_from segmentation --dataset_to_copy_to yoloseg

## For Stacking Images
python3 utils/stack_images.py --dataset yoloseg
python3 utils/copy_labels.py --dataset yoloseg

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

rm -r ./t1c_yoloseg
rm -r ./t1n_yoloseg
rm -r ./t2f_yoloseg
rm -r ./t2w_yoloseg
