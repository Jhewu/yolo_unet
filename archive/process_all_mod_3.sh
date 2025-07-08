#!/bin/bash

python3 utils/split_dataset_YOLO.py
python3 utils/brats_2d_slicer_YOLO.py
python3 utils/binarize_gt.py
# python3 utils/npy_to_png.py
python3 utils/copy_training_img.py

rm -r ./dataset_split
rm -r ./dataset_sliced
rm -r ./binarized_masks