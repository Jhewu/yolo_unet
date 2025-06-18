# YOLOv11-Seg for BraTS-PED

This repository uses YOLOv11-Seg to segment 2D scans from BraTS-PED

## Explanation of Files within the Directories
1. yolov11.py is the main file to run YOLOv11
2. parameters.py stores the parameters you would modify in order to run yolov11.py
3. callback/ is the directories where the specified callbacks would be stored when you run yolov11.py
4. runs/ stores all of the validation/testing results when you run YOLOv11.py
5. yolo11n-seg...dataset/ these folders contains the models trained specified in the write-up. You can find the weights, the hyperparameters, all the results, plots, and the images within this folder. 
6. datasets/ contains all of the 3 datasets mentioned in the write-up. There are 6 .yaml files. 3 of them are for training, and the other 3 are for testing. 
6. train_all.sh is a bash script that will train a YOLOn11-seg and a YOLOx11-seg model in the specified dataset you write within the bash script
7. yolo11n-..pt these are pretrained models used when traninig

## How to use
Before you start running yolov11-seg.py, you need to go to datasets, and modify the "path" from each of the .yaml files to your local working directory where this file is located. Otherwise yolov11-seg, would not be able to find the dataset. If your working directory is: /home/Desktop/YOLO11-Seg, make sure to add datasets/ and the dataset you're using after it, such as /home/Desktop/YOLO11-seg/datasets/all_modality_dataset. I was not able to find a way to automatically do this, since we are switching between computers/directories. 

You will be mainly using parameters.py to change hyperparameters. To run yolov11.py in its training mode, change the MODE to "train." After that, make sure to specify the MODEL you want to use, such as yolon11-seg or yolox11-seg, and specify the dataset you're using in DATASET such as "all_modality_dataset," this is important for training. The epoch is set to 100 with a batch of 16, and all of the other hyperparameters are set to default. 

To run all other modes, change MODE to "val", "test" and "predict." For all of these modes, you will be need to specify where the weights are in each of the respective variable such as BEST_MODEL_DIR_VAL, BEST_MODEL_DIR_TEST, BEST_MODEL_DIR_PREDICT. For predict, it's set to predict from the PNG image in the working directory, but you can change it by changing IMAGE_TO_TEST

