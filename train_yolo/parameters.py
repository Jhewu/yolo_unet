"""
All hyperparameters configured within this file. 
The other hyperparameters are in "default" mode, check 
https://docs.ultralytics.com/usage/cfg/#tasks
for more information
"""

## General parameters
MODE = "train"            # train, val, test, predict 
MODEL = "yolo11n"
DATASET = "data"
SEED = 42

## Training parameters
EPOCH = 30
BATCH = 512
IMAGE_SIZE = 192
CLOSE_MOSAIC = 0
FRACTION = 0.7

COS_LR = False
PROFILE=False
MULTI_SCALE = False
SINGLE_CLS = True
MIX_PRECISION = True
PLOT = True

FREEZE = None

# Loss Weights
CLS=0.5 
BOX=7.5 
DFL=1.5

## Augmentation parameters 
HSV_H = 0.0
HSV_S = 0.0
HSV_V = 0.25
DEGREES = 0.0
TRANSLATE = 0.025
SCALE = 0.0
FLIPUD = 0.5
FLIPLR = 0.5
MOSAIC = 0.5

LOAD_AND_TRAIN = False
BEST_MODEL_DIR_TRAIN = "train_yolo12n_2025_06_14_23_11_57/yolo12n_t1c_dataset/weights/best.pt"

"""Validation"""
BEST_MODEL_DIR_VAL = "yolo11n-seg_all_modality_dataset/weights/best.pt"

"""Testing"""
BEST_MODEL_DIR_TEST = "yolo11n-seg_all_modality_dataset/weights/best.pt"

"""Predict"""
BEST_MODEL_DIR_PREDICT = "yolo11n-seg_all_modality_dataset/weights/best.pt"
IMAGE_TO_TEST = "BraTS-PED-00003-00091-t1c.png"
