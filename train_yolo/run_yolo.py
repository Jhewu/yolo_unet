# Import Libraries
from ultralytics import YOLO
import torch 
import os
import csv
import cv2 as cv
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

    # get epoch
    epoch = trainer.epoch

    # get metric
    mAP = trainer.metrics["metrics/mAP50-95(M)"]

    # write csv_file to directory
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

""""Main Runtime"""
def RunYOLOv11Seg(mode):
    global CURRENT_TIME
    CURRENT_TIME = GetCurrentTime() 

    if mode == "train":
        print("/\nStarting training...")
       
        CreateDir("callbacks")

        if LOAD_AND_TRAIN: 
            model = YOLO(BEST_MODEL_DIR_TRAIN)
        else: 
            # load pretrained model (recommended for training)
            model = YOLO(f"{MODEL}.pt")

        # add callback for the model
        # model.add_callback("on_train_epoch_end", LogMetricMemorySpeed)

        print(f"\nThis is dataset {f"./datasets/{DATASET}.yaml"}\n")

        # train the model
        results = model.train(data=f"./datasets/{DATASET}.yaml", 
                              epochs=EPOCH, 
                              imgsz=IMAGE_SIZE, 
                              seed=SEED, 
                              batch=BATCH,
                              amp=MIX_PRECISION, 
                              plots=True,
                              project=f"{MODE}_{MODEL}_{CURRENT_TIME}",
                              name=f"{MODEL}_{DATASET}")
        print(f"\nFinish training, please check your directory for folder named 'train-....")
        
    elif mode == "val":
        print("/\nStarting validation...\n")
        print(f"\nFetching weights from...{BEST_MODEL_DIR_VAL}\n")
        model = YOLO(BEST_MODEL_DIR_VAL)
        metrics = model.val(plots=True, 
                            name=f"{MODE}_{MODEL}_{DATASET}")
        print(f"\nmAP50-95: {metrics.seg.map}\n")
        print(f"\nFinish validation, please check your directory for folder named 'val-....")

    elif mode == "test":
        print("\nStarting test...\n")
        print(f"\nFetching weights from...{BEST_MODEL_DIR_TEST}\n")
        model = YOLO(BEST_MODEL_DIR_TEST)
        metrics = model.val(data=f"./datasets/{DATASET}_test.yaml",
                            plots=True, 
                            name=f"{MODE}_{MODEL}_{DATASET}"
        )
        print(f"\nmAP50-95: {metrics.seg.map}\n")
        print(f"\nFinish testing, please check your directory for folder named 'test-....")
    
    elif mode == "predict":
        print("\nStarting prediction...\n")
        print(f"\nFetching weights from...{BEST_MODEL_DIR_PREDICT}\n")
        model = YOLO(BEST_MODEL_DIR_PREDICT)
        model.add_callback("on_predict_end", PrintMemoryUsed)

        results = model("BraTS-PED-00003-00091-t1c.png")
        
        # Save the prediction
        for result in results:
            result.save(filename="result.jpg")  # save to disk

        print(f"\nFinish prediction, please check your directory for a file named 'results.jpg'")

if __name__ == "__main__":
    RunYOLOv11Seg(MODE)

