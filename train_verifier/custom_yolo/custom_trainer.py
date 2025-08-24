from ultralytics.models.yolo.classify import ClassificationTrainer
from copy import copy

from custom_yolo.custom_classification_dataset import CustomClassificationDataset
from custom_yolo.custom_classification_validator import CustomClassificationValidator

class CustomClassificationTrainer(ClassificationTrainer): 
    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """
        Create a ClassificationDataset instance given an image path and mode.

        Args:
            img_path (str): Path to the dataset images.
            mode (str, optional): Dataset mode ('train', 'val', or 'test').
            batch (Any, optional): Batch information (unused in this implementation).

        Returns:
            (ClassificationDataset): Dataset for the specified mode.
        """
        return CustomClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)
    
    def get_validator(self):
        """Return an instance of ClassificationValidator for validation."""
        self.loss_names = ["loss"]
        return CustomClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

