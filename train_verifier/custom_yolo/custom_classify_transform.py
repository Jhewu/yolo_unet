import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, IterableSimpleNamespace, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xywh2xyxy, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0, 1.0)

def custom_classify_transforms(
    size: Union[Tuple[int, int], int] = 224,
    mean: Tuple[float, float, float] = DEFAULT_MEAN,
    std: Tuple[float, float, float] = DEFAULT_STD,
    interpolation: str = "BILINEAR",
    crop_fraction: float = None,
):
    """
    Create a composition of image transforms for classification tasks.

    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (Tuple[float, float, float]): Mean values for each RGB channel used in normalization.
        std (Tuple[float, float, float]): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float): Deprecated, will be removed in a future version.

    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    scale_size = size if isinstance(size, (tuple, list)) and len(size) == 2 else (size, size)

    if crop_fraction:
        raise DeprecationWarning(
            "'crop_fraction' arg of classify_transforms is deprecated, will be removed in a future version."
        )

    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # Resize the shortest edge to matching target dim for non-square target
        tfl = [T.Resize(scale_size)]
    tfl += [T.CenterCrop(size), T.ToTensor(), T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return T.Compose(tfl)