"""
All hyperparameters configured within
this file. The other hyperparameters
are in "default" mode, check 
https://docs.ultralytics.com/usage/cfg/#tasks
for more information
"""

"""General"""
MODE = "train"            # train, val, test, predict 
SEED = 42
MODEL = "yolo12m"

DATASET = "data"
                            # t1c_dataset for single modality
                            # all_modality_dataset for all modality
                            # stacked_dataset for three modality stacked into one image

"""Training"""
EPOCH = 100
BATCH = 164
IMAGE_SIZE = 192
MIX_PRECISION = True

LOAD_AND_TRAIN = False
BEST_MODEL_DIR_TRAIN = "train_yolo12n_2025_06_14_23_11_57/yolo12n_t1c_dataset/weights/best.pt"

"""Validation"""
BEST_MODEL_DIR_VAL = "yolo11n-seg_all_modality_dataset/weights/best.pt"

"""Testing"""
BEST_MODEL_DIR_TEST = "yolo11n-seg_all_modality_dataset/weights/best.pt"

"""Predict"""
BEST_MODEL_DIR_PREDICT = "yolo11n-seg_all_modality_dataset/weights/best.pt"
IMAGE_TO_TEST = "BraTS-PED-00003-00091-t1c.png"