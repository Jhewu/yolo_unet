#!/bin/bash

python3 split_dataset_YOLO.py
python3 brats_2d_slicer_YOLO.py
python3 binarize_gt.py
python3 npy_to_png.py
python3 copy_training_img.py

rm -r ./dataset_split
rm -r ./dataset_sliced
rm -r ./binarized_masks