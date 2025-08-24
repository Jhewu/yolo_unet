import os
import cv2
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def MaskToPolygons2(dir_to_examine, output_dir):
    CreateDir(output_dir)
    convert_segment_masks_to_yolo_seg(masks_dir=dir_to_examine, output_dir=output_dir, classes=1)

def Main(): 
    for split in ["test", "train", "val"]:
        label_dir = os.path.join("yolo_cropped_verified", "mask2", split)
        dest_dir = os.path.join("yolo_cropped_verified", "labels", split)
        MaskToPolygons2(label_dir, dest_dir)

if __name__ == "__main__": 
    # ------------------------------------------------------------------
    des="""
    Original script from computervisioneng on YouTube (slightly modified)

    An additional script that converts segmentation 
    masks (binary JPGs) into the YOLO segmentation polygons format
    """
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "."
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "."
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10

    Main()
    print("\nFinish converting binary mask to polygon, check your directory for labels")