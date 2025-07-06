#!/bin/bash

python3 split_dataset_YOLO.py
python3 brats_2d_slicer_YOLO.py
python3 binarize_gt.py
python3 masks_to_polygons.py
python3 copy_training_img.py

rm -r ./binarized_t1c
rm -r ./binarized_t1n
rm -r ./binarized_t2f
rm -r ./binarized_t2w

rm -r ./dataset_split

rm -r ./t1c
rm -r ./t1n
rm -r ./t2f
rm -r ./t2w