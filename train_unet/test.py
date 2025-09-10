from custom_predictor.custom_detection_predictor import CustomSegmentationPredictor
import os
MODEL_PATH = "/home/jun/Desktop/inspirit/yolo_unet/train_unet/train_yolo12n-seg_2025_08_27_01_01_59/yolo12n-seg_data/weights/best.pt"
# image = "BraTS-SSA-00041-00013-t1c.png"
image = os.path.join("stacked_segmentation/images/test/BraTS-SSA-00041-00036-t1c.png")

predictor = CustomSegmentationPredictor()
predictor.setup_model(MODEL_PATH)

results = predictor(image)
pred_mask = results[0].masks  # Masks object for segmentation masks outputs

print(pred_mask)


