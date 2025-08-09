from custom_yolo.custom_train import CustomDetectionTrainer
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

def ValYOLO(): 
    # elif mode == "val":
    #     print("/\nStarting validation...\n")
    #     print(f"\nFetching weights from...{BEST_MODEL_DIR_VAL}\n")
    #     model = YOLO(BEST_MODEL_DIR_VAL)
    #     results = model.val(plots=True, 
    #                         name=f"{MODE}_{MODEL}_{DATASET}")
    #     print("Class indices with average precision:", results.ap_class_index)
    #     print("Average precision for all classes:", results.box.all_ap)
    #     print("Average precision:", results.box.ap)
    #     print("Average precision at IoU=0.50:", results.box.ap50)
    #     print("Class indices for average precision:", results.box.ap_class_index)
    #     print("Class-specific results:", results.box.class_result)
    #     print("F1 score:", results.box.f1)
    #     print("F1 score curve:", results.box.f1_curve)
    #     print("Overall fitness score:", results.box.fitness)
    #     print("Mean average precision:", results.box.map)
    #     print("Mean average precision at IoU=0.50:", results.box.map50)
    #     print("Mean average precision at IoU=0.75:", results.box.map75)
    #     print("Mean average precision for different IoU thresholds:", results.box.maps)
    #     print("Mean results for different metrics:", results.box.mean_results)
    #     print("Mean precision:", results.box.mp)
    #     print("Mean recall:", results.box.mr)
    #     print("Precision:", results.box.p)
    #     print("Precision curve:", results.box.p_curve)
    #     print("Precision values:", results.box.prec_values)
    #     print("Specific precision metrics:", results.box.px)
    #     print("Recall:", results.box.r)
    #     print("Recall curve:", results.box.r_curve)
    #     print(f"\nFinish validation, please check your directory for folder named 'val-....")


    pass

def PredYOLO(): 

        # elif mode == "test":
    #     print("\nStarting test...\n")
    #     print(f"\nFetching weights from...{BEST_MODEL_DIR_TEST}\n")
    #     model = YOLO(BEST_MODEL_DIR_TEST)
    #     metrics = model.val(data=f"./datasets/{DATASET}_test.yaml",
    #                         plots=True, 
    #                         name=f"{MODE}_{MODEL}_{DATASET}"
    #     )
    #     print(f"\nmAP50-95: {metrics.seg.map}\n")
    #     print(f"\nFinish testing, please check your directory for folder named 'test-....")
    
    # elif mode == "predict":
    #     print("\nStarting prediction...\n")
    #     print(f"\nFetching weights from...{BEST_MODEL_DIR_PREDICT}\n")
    #     model = YOLO(BEST_MODEL_DIR_PREDICT)
    #     model.add_callback("on_predict_end", PrintMemoryUsed)

    #     results = model("BraTS-PED-00003-00091-t1c.png")
        
    #     # Save the prediction
    #     for result in results:
    #         result.save(filename="result.jpg")  # save to disk

    #     print(f"\nFinish prediction, please check your directory for a file named 'results.jpg'")

    pass

def TrainYOLO():
    print("\nStarting Training...")
    CreateDir("callbacks")

    # Add callback for the model
    # model.add_callback("on_train_epoch_end", LogMetricMemorySpeed)

    print(f"\nThis is dataset {f"./datasets/{DATASET}.yaml"}\n")

    args = dict(# General 
                model=f"{MODEL}.pt", 
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
        args["model"] = BEST_MODEL_DIR_TRAIN, args["resume"] = True
    
    trainer = CustomDetectionTrainer(overrides=args)
    trainer.train()

    print(f"\Ensuring the Model's input layer was changed: {trainer.setup_model()}")
    print(f"\nFinish training, please check your directory for folder named 'train-....")
        
if __name__ == "__main__":
    if MODE == "train": TrainYOLO()
    elif MODE == "val": ValYOLO()
    elif MODE == "predict": PredYOLO()
    else: print("\nPlease Configure a MODE in parameters.py...")
    

