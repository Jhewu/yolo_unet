from ultralytics import YOLO
from parameters import *
import time
import os

import torch

from custom_yolo.custom_trainer import CustomSegmentationTrainer

class YOLO_UNET(torch.nn.Module):
    def __init__(self, YOLO, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.YOLO = YOLO
        self.yolo_encoder = None 
        self.yolo_decoder = None
        self.bottleneck = None

    def load_yolo_encoder(self): 


        pass


    def forward(self):
        pass



def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def GetCurrentTime(): 
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

def Main(): 

    args = dict(
            ### General 
            model=BEST_MODEL_DIR_TRAIN, 
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
            seed=SEED, 
            batch=BATCH,
            amp=MIX_PRECISION, 
            multi_scale=MULTI_SCALE, 
            cos_lr=COS_LR,
            plots=PLOT,
            profile=PROFILE,
            project=f"{MODE}_{MODEL}_{GetCurrentTime()}",
            name=f"{MODEL}_{DATASET}", 
            
            ### Data Augmentation
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
    
    trainer = CustomSegmentationTrainer(overrides=args)


    pass

if __name__ == "__main__": 
    Main()