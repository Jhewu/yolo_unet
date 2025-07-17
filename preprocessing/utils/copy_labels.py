import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name)   

def CopyTree(src, dst):
    # Ensure the source exists
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory '{src}' does not exist.")
    try:
        # Recursively copy the directory tree
        shutil.copytree(src, dst, dirs_exist_ok=True)  # dirs_exist_ok=True to allow copying into an existing directory
        print(f"Successfully copied {src} to {dst}")
    except Exception as e:
        print(f"Error occurred while copying: {e}")

def CopyLabels(): 
    input_folder = f"{IN_DIR}/t1c_{DATASET}/labels"
    dest_folder = f"{OUT_DIR}/stacked_{DATASET}/labels"
    CreateDir(dest_folder)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            executor.submit(CopyTree, input_folder, dest_folder)

if __name__ == "__main__": 
    # -------------------------------------------------------------
    des="""
    As the name suggest, this copies the training images
    to the desired directory.
    """
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    parser.add_argument('--dataset',type=str,help='options are detection, segmentation, yoloseg\t[None]')
    args = parser.parse_args()

    if args.dataset is not None:
        DATASET = args.dataset
    else: DATASET = "detection"
    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "."
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = "."
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10

    CopyLabels()
    print("\nFinish copying, please check your directory\n")
