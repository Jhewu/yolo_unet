from custom_predictor.custom_detection_predictor import CustomDetectionPredictor
import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor

import argparse
import os

import piexif
import cv2
from PIL import Image

image = "BraTS-SSA-00041-00013-t1c.png"

predictor = Custom