

from typing import Tuple

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0, 1.0)


def custom_classify_augmentations(
    size: int = 224,
    mean: Tuple[float, float, float] = DEFAULT_MEAN,
    std: Tuple[float, float, float] = DEFAULT_STD,
    scale: Tuple[float, float] = None,
    ratio: Tuple[float, float] = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    auto_augment: str = None,
    hsv_h: float = 0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s: float = 0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v: float = 0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter: bool = False,
    erasing: float = 0.0,
    interpolation: str = "BILINEAR",
):
    """
    Create a composition of image augmentation transforms for classification tasks.

    This function generates a set of image transformations suitable for training classification models. It includes
    options for resizing, flipping, color jittering, auto augmentation, and random erasing.

    Args:
        size (int): Target size for the image after transformations.
        mean (Tuple[float, float, float]): Mean values for each RGB channel used in normalization.
        std (Tuple[float, float, float]): Standard deviation values for each RGB channel used in normalization.
        scale (Tuple[float, float] | None): Range of size of the origin size cropped.
        ratio (Tuple[float, float] | None): Range of aspect ratio of the origin aspect ratio cropped.
        hflip (float): Probability of horizontal flip.
        vflip (float): Probability of vertical flip.
        auto_augment (str | None): Auto augmentation policy. Can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): Image HSV-Hue augmentation factor.
        hsv_s (float): Image HSV-Saturation augmentation factor.
        hsv_v (float): Image HSV-Value augmentation factor.
        force_color_jitter (bool): Whether to apply color jitter even if auto augment is enabled.
        erasing (float): Probability of random erasing.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.

    Returns:
        (torchvision.transforms.Compose): A composition of image augmentation transforms.

    Examples:
        >>> transforms = classify_augmentations(size=224, auto_augment="randaugment")
        >>> augmented_image = transforms(original_image)
    """
    # Transforms to apply if Albumentations not installed
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_augmentations() size {size} must be integer, not (list, tuple)")
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str), f"Provided argument should be string, but got type {type(auto_augment)}"
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)