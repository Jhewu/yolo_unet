"""
All hyperparameters configured within this file. 
The other hyperparameters are in "default" mode, check 
https://docs.ultralytics.com/usage/cfg/#tasks
for more information
"""

## General parameters
MODE = "train"            # train, val, test, predict 
MODEL = "yolo12x-cls"
DATASET = "/home/jun/Desktop/inspirit/yolo_unet/train_verifier/ssa_classification_dataset_"
SEED = 42

## Training parameters
PRETRAINED = False
RESUME = False
EPOCH = 50
BATCH = 128
IMAGE_SIZE = 192
FRACTION = 1.0
INITIAL_LR = 1e-5
FINAL_LR = 1e-5
WARMUP_EPOCH = 10

COS_LR = True
PROFILE = False
MULTI_SCALE = True
MIX_PRECISION = True
PLOT = True

FREEZE = 0

## Augmentation parameters 
HSV_H = 0.0
HSV_S = 0.0
MOSAIC = 0.0

HSV_V = 0.25
TRANSLATE = 0.1
SCALE = 0.25
FLIPUD = 0.5
FLIPLR = 0.5

DEGREES = 0.1           # 2.5 use it carefully
SHEAR =  1              # 5 use it carefully
PERSPECTIVE = 0.001     # 0.010 use it carefully
MIXUP = 0.0             # 0.5 maybe a good and also bad idea
CUTMIX = 0.0            # maybe but l

LOAD_AND_TRAIN = True
BEST_MODEL_DIR_TRAIN = "/home/jun/Desktop/inspirit/yolo_unet/train_verifier/train_yolo12n-cls_gli_verifier_dataset_n/yolo12n-cls_/home/jun/Desktop/inspirit/yolo_unet/train_verifier/gli_verifier_dataset_n/weights/best.pt"
# BEST_MODEL_DIR_TRAIN = "/home/jun/Desktop/inspirit/yolo_unet/train_yolo/train_yolo12s-cls_2025_08_23_19_11_31/yolo12s-cls_/home/jun/Desktop/inspirit/yolo_unet/train_yolo/verifier_dataset/weights/last.pt"



