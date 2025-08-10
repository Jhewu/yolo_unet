"""
All hyperparameters configured within this file. 
The other hyperparameters are in "default" mode, check 
https://docs.ultralytics.com/usage/cfg/#tasks
for more information
"""

## General parameters
MODE = "train"            # train, val, test, predict 
MODEL = "yolo11s"
DATASET = "data"
SEED = 42

## Training parameters
PRETRAINED = False
RESUME = False
EPOCH = 50
BATCH = 512
IMAGE_SIZE = 192
CLOSE_MOSAIC = 0
FRACTION = 1.0
INITIAL_LR = 1e-8
FINAL_LR = 1e-8
WARMUP_EPOCH = 10

COS_LR = True
PROFILE = False
MULTI_SCALE = False
SINGLE_CLS = True
MIX_PRECISION = True
PLOT = True

FREEZE = 10

# Loss Weights
CLS=0.5 
BOX=6.5 # 7.5 
DFL=2.5 # 1.5

## Augmentation parameters 
HSV_H = 0.0
HSV_S = 0.0
MOSAIC = 0.0

HSV_V = 0.0 # 0.25, previously, but with 4-channels, it does not work anymore
TRANSLATE = 0.1
SCALE = 0.25
FLIPUD = 0.5
FLIPLR = 0.5

DEGREES = 0.1 # 2.5 use it carefully
SHEAR =  1 # 5 use it carefully
PERSPECTIVE = 0.001 # 0.010 use it carefully
MIXUP = 0.0 # 0.5 maybe a good and also bad idea
CUTMIX = 0.0 # maybe but l

LOAD_AND_TRAIN = True
BEST_MODEL_DIR_TRAIN = "/home/jun/Desktop/inspirit/yolo_unet/train_yolo/train_yolo11s_2025_08_08_23_48_29/yolo11s_data/weights/best.pt"


"""Validation"""
BEST_MODEL_DIR_VAL = "yolo11n-seg_all_modality_dataset/weights/best.pt"

"""Testing"""
BEST_MODEL_DIR_TEST = "yolo11n-seg_all_modality_dataset/weights/best.pt"

"""Predict"""
BEST_MODEL_DIR_PREDICT = "yolo11n-seg_all_modality_dataset/weights/best.pt"
IMAGE_TO_TEST = "BraTS-PED-00003-00091-t1c.png"


