"""
All hyperparameters configured within this file. 
The other hyperparameters are in "default" mode, check 
https://docs.ultralytics.com/usage/cfg/#tasks
for more information
"""

### General parameters
MODE = "train"            # train, val, test, predict 
MODEL = "yolo12n-seg"
DATASET = "data"
SEED = 42

### Training parameters
PRETRAINED = False
RESUME = True
EPOCH = 100
BATCH = 128
IMAGE_SIZE = 160
CLOSE_MOSAIC = 0
FRACTION = 1.0
INITIAL_LR = 1e-5
FINAL_LR = 1e-5
WARMUP_EPOCH = 10

COS_LR = True
PROFILE = False
MULTI_SCALE = False
SINGLE_CLS = True
MIX_PRECISION = True
PLOT = True

FREEZE = 0

# Loss Weights
# CLS=0.5 
# BOX=6       # 6.5 # 7.5 
# DFL=3.5     # 2.5 # 1.5

### Augmentation parameters 
HSV_H = 0.0
HSV_S = 0.0
MOSAIC = 0.0

HSV_V = 0.0         # 0.25, previously, but with 4-channels, it does not work anymore
TRANSLATE = 0.1
SCALE = 0.25
FLIPUD = 0.5
FLIPLR = 0.5

DEGREES = 0.1       # 2.5 use it carefully
SHEAR =  1          # 5 use it carefully
PERSPECTIVE = 0.001 # 0.010 use it carefully
MIXUP = 0.0         # 0.5 maybe a good and also bad idea
CUTMIX = 0.0        # maybe 

LOAD_AND_TRAIN = True
# BEST_MODEL_DIR_TRAIN = "/home/jun/Desktop/inspirit/yolo_unet/train_yolo/runs/trained_gli_checkpoint/yolo11s_data/weights/best.pt"
BEST_MODEL_DIR_TRAIN = "/home/jun/Desktop/inspirit/yolo_unet/train_yolo/train_yolo12n-seg_2025_08_30_01_05_37/yolo12n-seg_data/weights/best.pt"

### Validation
BEST_MODEL_DIR_VAL = ""

### Testing
BEST_MODEL_DIR_TEST = ""

### Predict
BEST_MODEL_DIR_PREDICT = ""
IMAGE_TO_TEST = ""

