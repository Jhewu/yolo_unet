from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images

from ultralytics.models.yolo.classify import ClassificationValidator
from custom_yolo.custom_classification_dataset import CustomClassificationDataset

class CustomClassificationValidator(ClassificationValidator):    
    def build_dataset(self, img_path: str) -> ClassificationDataset:
        """Create a ClassificationDataset instance for validation."""
        return CustomClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)