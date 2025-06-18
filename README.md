# YOLO+UNET-like Ensemble Architecture for BraTS-SSA

This repository contains an implementation of a VRAM efficient and fast Brain Tumor Segmentation Model for the BraTS-SSA 2025 Dataset

### The repository is organized as follow
```
├── data_sample/   (sample BraTS-SSA slices)
├── preprocessing/ (preprocess the 3D BraTS-SSA into 2D)
├── train_data/    (stores the processed training data)
├── train_unet/    (stores UNET training files)
├── train_yolo/    (stores YOLO training files)
├── yolo_weights/  (stores the YOLO-weights for yolo_crop.py)
```