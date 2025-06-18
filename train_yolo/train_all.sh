#!/bin/bash

# Modify parameters in parameters.py using sed
# New value in the end

# run small model
sed -i 's/^MODEL = .*/MODEL = "yolo11n-seg"/' parameters.py
sed -i 's/^DATASET = .*/DATASET = "t1c_dataset"/' parameters.py
python3 yolov11.py

# run big model
sed -i 's/^MODEL = .*/MODEL = "yolo11x-seg"/' parameters.py
sed -i 's/^DATASET = .*/DATASET = "t1c_dataset"/' parameters.py
python3 yolov11.py

sleep 1200

./train_all2.sh