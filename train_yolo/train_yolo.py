# from custom_yolo.custom_train import CustomDetectionTrainer
from ultralytics import RTDETR
from ultralytics import YOLO
import torch.nn as nn
import torch 
import os
import csv
import time

"""
TO CHANGE HYPERPARAMETERS GO TO PARAMETERS.PY
"""
from parameters import *

TIME = 0
CURRENT_TIME = 0

"""Helper Functions"""
def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def GetCurrentTime(): 
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

"""Define Custom Callbacks"""
def PrintMemoryUsed(predictor):
    memory_used = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"\nThis is memory used: {memory_used}")

def LogMetricMemorySpeed(trainer): 
    global TIME

    # get time
    if trainer.epoch_time is not None:
        TIME += trainer.epoch_time

    # get the current GPU memory usage (in MB)
    memory_used = torch.cuda.memory_allocated() / (1024 ** 2)

    epoch = trainer.epoch
    mAP = trainer.metrics["metrics/mAP50-95(M)"]

    # Write csv_file to directory
    data = [{'epoch': epoch, 'mAP50-95': mAP, 'time': TIME, 'memory': memory_used}]
    CreateDir(f"callbacks/{CURRENT_TIME}")
    callback_dir = f"callbacks/{CURRENT_TIME}/csv_callbacks_{DATASET}_{MODEL}_{MODE}.csv"
    with open(callback_dir, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'mAP50-95', 'time', 'memory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvfile.seek(0, 2) 
        if csvfile.tell() == 0: 
            writer.writeheader()
        writer.writerows(data)

def TrainYOLO():
    print("\nStarting Training...")
    CreateDir("callbacks")

    # Add callback for the model
    # model.add_callback("on_train_epoch_end", LogMetricMemorySpeed)

    print(f"\nThis is dataset {f"./datasets/{DATASET}.yaml"}\n")

    args = dict(# General 
                model=f"{MODEL}.yaml", 
                data=f"datasets/{DATASET}.yaml", 
                epochs=EPOCH, 
                pretrained=PRETRAINED, 
                imgsz=IMAGE_SIZE, 
                single_cls=SINGLE_CLS, 
                close_mosaic=CLOSE_MOSAIC, 
                fraction=FRACTION,
                freeze=FREEZE,  
                lr0=INITIAL_LR, 
                lrf=FINAL_LR, 
                warmup_epochs=WARMUP_EPOCH, 
                cls=CLS, 
                box=BOX, 
                dfl=DFL, 
                seed=SEED, 
                batch=BATCH,
                amp=MIX_PRECISION, 
                multi_scale=MULTI_SCALE, 
                cos_lr=COS_LR,
                plots=PLOT,
                profile=PROFILE,
                project=f"{MODE}_{MODEL}_{GetCurrentTime()}",
                name=f"{MODEL}_{DATASET}", 
                
                # Data Augmentation
                hsv_h=HSV_H, 
                hsv_s=HSV_S, 
                hsv_v=HSV_V, 
                degrees=DEGREES,
                translate=TRANSLATE,
                scale=SCALE,
                flipud=FLIPUD, 
                fliplr=FLIPLR, 
                mosaic=MOSAIC, 
                shear=SHEAR, 
                perspective=PERSPECTIVE, 
                mixup=MIXUP, 
                cutmix=CUTMIX)
    
    if LOAD_AND_TRAIN: 
        print("\nLoading and Training...")
        args["model"] = BEST_MODEL_DIR_TRAIN
        args["resume"] = RESUME

    model = RTDETR("rtdetr-l.pt")
              
    model.train(
          data=f"datasets/{DATASET}.yaml", 
                epochs=EPOCH, 
                pretrained=PRETRAINED, 
                imgsz=IMAGE_SIZE, 
                single_cls=SINGLE_CLS, 
                close_mosaic=CLOSE_MOSAIC, 
                fraction=FRACTION,
                freeze=FREEZE,  
                lr0=INITIAL_LR, 
                lrf=FINAL_LR, 
                warmup_epochs=WARMUP_EPOCH, 
                cls=CLS, 
                box=BOX, 
                dfl=DFL, 
                seed=SEED, 
                batch=BATCH,
                amp=MIX_PRECISION, 
                multi_scale=MULTI_SCALE, 
                cos_lr=COS_LR,
                plots=PLOT,
                profile=PROFILE,
                project=f"{MODE}_{MODEL}_{GetCurrentTime()}",
                name=f"{MODEL}_{DATASET}", 
                
                # Data Augmentation
                hsv_h=HSV_H, 
                hsv_s=HSV_S, 
                hsv_v=HSV_V, 
                degrees=DEGREES,
                translate=TRANSLATE,
                scale=SCALE,
                flipud=FLIPUD, 
                fliplr=FLIPLR, 
                mosaic=MOSAIC, 
                shear=SHEAR, 
                perspective=PERSPECTIVE, 
                mixup=MIXUP, 
                cutmix=CUTMIX)

                   
    # trainer = CustomDetectionTrainer(overrides=args)
    # trainer.train()

    # print(f"\nEnsuring the Model's input layer was changed: {trainer.setup_model()}")
    print(f"\nFinish training, please check your directory for folder named 'train-....")
        
if __name__ == "__main__":
    if MODE == "train": TrainYOLO()

    else: print("\nPlease Configure a MODE in parameters.py...")
