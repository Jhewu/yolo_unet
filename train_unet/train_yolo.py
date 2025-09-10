from ultralytics import YOLO
from custom_yolo.custom_trainer import CustomSegmentationTrainer
from parameters import *
import time
import os

def GetCurrentTime(): 
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)  

# def train_yolo(): 
    
#     if LOAD_AND_TRAIN: 
#         model = YOLO(BEST_MODEL_DIR_TRAIN)
#     else: model = YOLO(f"{MODEL}.yaml")

#     model.train(
#                 data="data.yaml",
#                 epochs=EPOCH, 
#                 pretrained=PRETRAINED, 
#                 imgsz=IMAGE_SIZE, 
#                 close_mosaic=CLOSE_MOSAIC, 
#                 fraction=FRACTION,
#                 freeze=FREEZE,  
#                 lr0=INITIAL_LR, 
#                 lrf=FINAL_LR, 
#                 warmup_epochs=WARMUP_EPOCH, 
#                 cls=CLS, 
#                 box=BOX, 
#                 dfl=DFL, 
#                 seed=SEED, 
#                 batch=BATCH,
#                 amp=MIX_PRECISION, 
#                 multi_scale=MULTI_SCALE, 
#                 cos_lr=COS_LR,
#                 plots=PLOT,
#                 profile=PROFILE,
#                 project=f"{MODE}_{MODEL}_{GetCurrentTime()}",
#                 name=f"{MODEL}_{DATASET}", 
                
#                 # Data Augmentation
#                 hsv_h=HSV_H, 
#                 hsv_s=HSV_S, 
#                 hsv_v=HSV_V, 
#                 degrees=DEGREES,
#                 translate=TRANSLATE,
#                 scale=SCALE,
#                 flipud=FLIPUD, 
#                 fliplr=FLIPLR, 
#                 mosaic=MOSAIC, 
#                 shear=SHEAR, 
#                 perspective=PERSPECTIVE, 
#                 mixup=MIXUP, 
#                 cutmix=CUTMIX)

#     # print(f"\nEnsuring the Model's input layer was changed: {trainer.setup_model()}")
#     print(f"\nFinish training, please check your directory for folder named 'train-....")

def train_yolo(): 
    print("\nStarting Training...")
    CreateDir("callbacks")

    # Add callback for the model
    # model.add_callback("on_train_epoch_end", LogMetricMemorySpeed)

    print(f"\nThis is dataset {f"./datasets/{DATASET}.yaml"}\n")

    args = dict(# General 
                model=f"{MODEL}.yaml", 
                data=f"{DATASET}.yaml", 
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
    
    trainer = CustomSegmentationTrainer(overrides=args)
    trainer.train()

    print(f"\nEnsuring the Model's input layer was changed: {trainer.setup_model()}")
    print(f"\nFinish training, please check your directory for folder named 'train-....")

if __name__ == "__main__":
    train_yolo()