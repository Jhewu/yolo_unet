"""
All hyperparameters configured within this file. 
The other hyperparameters are in "default" mode, check 
https://docs.ultralytics.com/usage/cfg/#tasks
for more information
"""

## General parameters
MODE = "train"            # train, val, test, predict 
MODEL = "yolo11_4ch"
DATASET = "data"
SEED = 42

## Training parameters
PRETRAINED = False
EPOCH = 5
BATCH = 512
IMAGE_SIZE = 192
CLOSE_MOSAIC = 0
FRACTION = 1.0
INITIAL_LR = 1e-5
FINAL_LR = 1e-5
WARMUP_EPOCH = 3

COS_LR = True
PROFILE = False
MULTI_SCALE = False
SINGLE_CLS = True
MIX_PRECISION = True
PLOT = True

FREEZE = 0 # 10

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

LOAD_AND_TRAIN = False
BEST_MODEL_DIR_TRAIN = "weights/best.pt"

"""Validation"""
BEST_MODEL_DIR_VAL = "yolo11n-seg_all_modality_dataset/weights/best.pt"

"""Testing"""
BEST_MODEL_DIR_TEST = "yolo11n-seg_all_modality_dataset/weights/best.pt"

"""Predict"""
BEST_MODEL_DIR_PREDICT = "yolo11n-seg_all_modality_dataset/weights/best.pt"
IMAGE_TO_TEST = "BraTS-PED-00003-00091-t1c.png"


#                    from  n    params  module                                       arguments                     

#   0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 

#   1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                

#   2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     

#   3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              

#   4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    

#   5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              

#   6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           

#   7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              

#   8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           

#   9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 

#  10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 

#  11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          

#  12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           

#  13                  -1  1    443776  ultralytics.nn.modules.block.C3k2            [768, 256, 1, False]          

#  14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          

#  15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           

#  16                  -1  1    127680  ultralytics.nn.modules.block.C3k2            [512, 128, 1, False]          

#  17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              

#  18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           

#  19                  -1  1    345472  ultralytics.nn.modules.block.C3k2            [384, 256, 1, False]          

#  20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              

#  21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           

#  22                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           

#  23        [16, 19, 22]  1    819795  ultralytics.nn.modules.head.Detect           [1, [128, 256, 512]]          