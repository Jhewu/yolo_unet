import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from ultralytics import YOLO

# Install Albumentations if you haven't already:
# pip install albumentations opencv-python

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CustomYOLODataset(Dataset):
    def __init__(self, img_paths, labels, img_size=640, augment=True):
        self.img_paths = img_paths
        self.labels = labels  # List of label tensors/arrays for each image
        self.img_size = img_size
        self.augment = augment

        # Define your custom Albumentations transform pipeline
        if self.augment:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size, p=1.0), # Resize to img_size while maintaining aspect ratio
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0), # Pad if needed
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2), # Example: add vertical flip for brain images
                A.Rotate(limit=20, p=0.7), # Rotate up to 20 degrees
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.2),
                # Add more custom augmentations relevant to your brain tumor segmentation
                # e.g., elastic deformations, random resized crops if applicable to segmentation
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet normalization (adjust for your data)
                ToTensorV2(), # Converts to PyTorch tensor and moves channel to front
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) # Crucial for bounding box transforms
        else:
            # For validation/testing, usually just resize and normalize
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size, p=1.0),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        labels = self.labels[idx] # Labels should be in YOLO format [class_id, x_center, y_center, width, height]

        # Load image (OpenCV loads in BGR, Albumentations expects RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Separate class labels from bounding box coordinates for Albumentations
        class_labels = labels[:, 0]
        bboxes = labels[:, 1:]

        # Apply transformations
        augmented = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        img_transformed = augmented['image']
        bboxes_transformed = augmented['bboxes']
        class_labels_transformed = augmented['class_labels']

        # Recombine class labels and bboxes into the YOLO format
        if len(bboxes_transformed) > 0:
            labels_transformed = np.concatenate((np.array(class_labels_transformed).reshape(-1, 1), np.array(bboxes_transformed)), axis=1)
        else:
            labels_transformed = np.zeros((0, 5), dtype=np.float32) # Handle case with no objects

        return img_transformed, torch.from_numpy(labels_transformed)

# Example usage (dummy data)
# In a real scenario, you would load your actual image paths and labels
dummy_img_paths = [f'path/to/image_{i}.jpg' for i in range(100)]
# Dummy labels: [class_id, x_center, y_center, width, height] normalized
dummy_labels = [
    np.array([[0, 0.5, 0.5, 0.2, 0.2], [1, 0.3, 0.3, 0.1, 0.1]], dtype=np.float32) for _ in range(50)
] + [
    np.array([[0, 0.7, 0.7, 0.15, 0.15]], dtype=np.float32) for _ in range(50)
] # Simulate some images with 2 objects, some with 1

# Create your custom datasets
train_dataset = CustomYOLODataset(dummy_img_paths[:80], dummy_labels[:80], augment=True)
val_dataset = CustomYOLODataset(dummy_img_paths[80:], dummy_labels[80:], augment=False)

# YOLOv8's `train` method expects a data.yaml file, but you can override the dataset/dataloader
# This requires a bit more advanced usage if you don't want to use the .yaml file
# The more common approach is to let YOLOv8 handle the DataLoader creation, but define
# how it loads and augments data internally by patching/extending their dataset class.

# Option 1: Directly use your custom DataLoader (more manual, but flexible)
# This requires you to create your own training loop or adapt YOLOv8's internal one
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))

# Option 2: Integrate with Ultralytics YOLO's `train` method (recommended)
# This involves defining your dataset as part of their internal dataset pipeline.
# Ultralytics models typically have a 'data' argument in model.train() that points to a YAML file.
# Inside that YAML, you can specify your custom dataset. This is usually done by
# either modifying their internal dataset creation logic (which they try to avoid)
# or by directly injecting a custom dataset object into the `Trainer`.

# More practically, you'd typically define a custom YAML for your dataset.
# Ultralytics allows you to control augmentations through hyperparameters in the `train` method
# or by modifying the `data/hyps/hyp.scratch-low.yaml` (or similar) file.
# However, for truly *custom* functions not covered by their default hyperparameters,
# you'd need to hook into their dataset processing.

# As of recent Ultralytics versions, directly swapping the entire DataLoader via a simple API call
# in `model.train()` is not straightforward for custom augmentations that modify the labels
# (like object detection bounding boxes). The most robust way without modifying source
# is to use Albumentations' integration or to override their `Dataset` class's `__getitem__`
# or `build_transforms` method.

# **The best API-based way to integrate a completely custom augmentation pipeline (like Albumentations
# with custom transforms)** is to leverage Albumentations, which Ultralytics *does* support
# and provide hooks for.

# **Example of integrating Albumentations (closer to how Ultralytics expects it):**

# First, ensure you have Albumentations installed.
# Ultralytics often uses Albumentations internally.
# You can define your Albumentations pipeline and then let YOLO use it.

# Define your Albumentations transform pipeline
custom_alb_transforms = A.Compose([
    A.LongestMaxSize(max_size=640, p=1.0),
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.2),
    # Add more, e.g., A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) # Essential for bbox transforms

# You then create a custom `data.yaml` that points to your images/labels
# and potentially a custom `hyp.yaml` if you want to control default Ultralytics augmentations.

# If you want to use a completely custom `__getitem__` logic that is not just
# an Albumentations pipeline, you would need to:
# 1. Inherit from `ultralytics.data.dataset.YOLODataSet` (or the relevant class for your task).
# 2. Override the `__getitem__` method to implement your custom loading and augmentation.
# 3. Potentially modify the `build_dataloader` function or the `Trainer` class to use your
#    custom `Dataset` directly. This often involves a small amount of "patching" or
#    extending their internal classes, which is still API-based but might feel
#    closer to modifying source.

# Example of extending their `Dataset` (less common for simple augmentation, more for custom data formats):
from ultralytics.data.dataset import YOLODataset, build_dataloader
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer # Or SegmentationTrainer

class MyCustomYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can override or extend self.transform here.
        # self.transform is typically a Compose of Albumentations transforms.
        # If you want a completely different pipeline, replace it.
        # If you want to add to it, you can append/insert.
        # For example, to add an elastic transform:
        self.transform.transforms.insert(2, A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5))


# Then, when you train, you might need to instantiate a custom `Trainer` that uses your dataset:
# from ultralytics import YOLO
# from ultralytics.models.yolo.detect import DetectionTrainer
# from ultralytics.nn.tasks import DetectionModel

# class CustomTrainer(DetectionTrainer):
#     def get_dataloader(self, batch_size, rank=0, mode='train'):
#         # This is where you can inject your custom dataset
#         # You'll need to pass your custom dataset class to the build_dataloader function
#         # This can get a bit complex as you need to match their expected input formats
#         # for labels, etc.
#         pass # This would involve calling build_dataloader with your custom dataset type

# The most straightforward way to add custom Albumentations transforms is to just define your `A.Compose`
# pipeline and ensure Ultralytics picks it up. While Ultralytics has default augmentation parameters
# you can set in `train()` or a `hyp.yaml` file, for truly custom `Albumentations` chains, you might
# need to:
# 1. **Modify `ultralytics/data/augment.py` directly if you want a quick hack (not recommended for clean projects).**
# 2. **Override `YOLODataset.build_transforms` or `YOLODataset.__getitem__`:** This is the more "API-based but advanced" way. You would subclass `YOLODataset` and replace the relevant method that constructs the augmentation pipeline.

**Conclusion:**

* **For common augmentations (flips, rotations, scaling, brightness/contrast, etc.), you can simply adjust hyperparameters in your `train` command or `hyp.yaml` file.** This is the simplest API usage.
* **For custom augmentations, especially those from `Albumentations` that modify both images and bounding boxes (like elastic transforms), you can define a custom `A.Compose` pipeline.** The challenge lies in ensuring Ultralytics' internal data loader uses *your* custom pipeline instead of its default.
    * The most practical "API-based" way without modifying Ultralytics' core files is to define your custom `Albumentations.Compose` and then **patch or replace the `transform` attribute of the `YOLODataset` object *after* it's been initialized by Ultralytics, but before training starts.** This is a bit of a runtime hack but avoids source code modification.
    * A cleaner but more involved API approach is to create a custom `YOLODataset` subclass and override its `build_transforms` or `__getitem__` method to incorporate your custom logic. Then, you would need to tell the `Trainer` to use your custom dataset class, which can sometimes require modifying the `Trainer` class itself.

**Simplified API Example (for Albumentations, generally supported by Ultralytics):**

Ultralytics is designed to be highly configurable. While direct injection of an entirely custom `DataLoader` into `model.train()` isn't a one-liner for custom transforms *that modify labels*, you can often achieve your goals by understanding how Ultralytics sets up its internal `Albumentations` pipeline. They provide a `build_transforms` function.

If your custom augmentations are based on `Albumentations`, the cleanest approach is often to:
1.  Define your desired `Albumentations.Compose` pipeline.
2.  If Ultralytics provides a hook to inject a custom `Albumentations` pipeline (they sometimes do in newer versions or through specific hyperparams), use that.
3.  If not, you might have to temporarily "monkey patch" the `build_transforms` function or the `dataset.transforms` attribute right before `model.train()` is called, for example:

```python
import torch
from ultralytics import YOLO
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

# Load a model
model = YOLO('yolov8n.pt')

# --- Define your custom Albumentations pipeline ---
# This is your custom augmentation logic for training
custom_train_transforms = A.Compose([
    A.LongestMaxSize(max_size=640, p=1.0),
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.7, interpolation=cv2.INTER_LINEAR),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.2),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0), # Your specific warp
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)), # Assuming pixel values are already 0-1 or will be handled by YOLO's default (img/255.0)
                                                              # For brain scans, you might need specific normalization or leave it to YOLO's default
    ToTensorV2(p=1.0),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# --- Option: Override the dataset's transforms before training ---
# This often requires access to the internal `YOLODataset` class.
# This approach might be less stable across Ultralytics versions as it relies on internal structure.
from ultralytics.data.dataset import YOLODataset, build_transforms

# You can modify the `build_transforms` function's behavior
# This is a bit more advanced and involves understanding Ultralytics internals.
# A simpler way, if your YAML is set up, is to pass your augmentations directly
# if there's an `augmentations` hyperparameter for `model.train()` or similar.

# For segmentation, the process is similar, ensuring masks are transformed alongside images.
# Albumentations handles this well if you pass `masks` along with `image` and `bboxes`.

# The most common way to influence augmentations without source code modification in YOLOv8
# is through the `hyp.yaml` file (hyperparameters) which controls default augmentations.
# For *truly* custom Albumentations transforms not covered by hyperparams,
# the Ultralytics maintainer (Glenn Jocher) has previously suggested in GitHub issues
# to create a custom `Dataset` class inheriting from `torch.utils.data.Dataset` and
# implementing your `__getitem__` with your custom transforms, then manually integrating
# this into the training loop if `model.train()` doesn't directly support it.

# **However, a recent discussion on Ultralytics GitHub (Issue #10174)
# suggests that for Albumentations, you can define your custom `A.Compose` and
# then integrate it within a custom `Dataset` class as shown in the first
# `CustomYOLODataset` example above.**
# Then, you would need to adjust the training script to use your `CustomYOLODataset`
# instead of their default one, which involves some modification to the training script,
# but not the core Ultralytics library.

# Example of how you *might* call train with a custom data loader, if Ultralytics directly supported it
# (This exact syntax might not work out-of-the-box, it illustrates the concept):
# results = model.train(
#     data='your_data.yaml', # Your data config
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     # This is illustrative. The actual API for custom transforms might be different
#     # or require subclassing their Trainer/Dataset.
#     # For Albumentations, look for a 'transforms' or 'augmentation_pipeline' argument.
#     # Currently, direct injection of a custom A.Compose into model.train() is not a direct argument.
# )

# The most robust API-based method:
# 1. Create a `CustomYOLODataset` (as shown at the top) that loads your data and applies your specific Albumentations.
# 2. Modify the `train.py` script (or your custom training script) to use your `CustomYOLODataset`
#    when constructing the `DataLoader`. This usually means replacing the default `YOLODataset`
#    initialization with your `CustomYOLODataset` initialization.

This requires you to have a good understanding of the Ultralytics training pipeline and PyTorch `Dataset`/`DataLoader` mechanics.