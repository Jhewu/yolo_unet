from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import de_parallel
from custom_yolo.custom_build_data import build_yolo_dataset
from ultralytics.utils import RANK
from typing import Optional

from ultralytics.data import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

class CustomSegmentationTrainer(SegmentationTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)