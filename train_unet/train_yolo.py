from ultralytics import YOLO
from parameters import *
import time

def GetCurrentTime(): 
    current_time = time.localtime()
    return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

def train_yolo(): 
    if LOAD_AND_TRAIN: 
        model = YOLO(BEST_MODEL_DIR_TRAIN)
    else: model = YOLO(f"{MODEL}.yaml")
    model.train(
                data="data.yaml",
                epochs=EPOCH, 
                pretrained=PRETRAINED, 
                imgsz=IMAGE_SIZE, 
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

    # print(f"\nEnsuring the Model's input layer was changed: {trainer.setup_model()}")
    print(f"\nFinish training, please check your directory for folder named 'train-....")

if __name__ == "__main__":
    train_yolo()