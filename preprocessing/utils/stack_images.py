import os
import cv2
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def CreateDir(folder_name):
   if not os.path.exists(folder_name):
       os.makedirs(folder_name) 

def CombinedStack(images): 
    return np.stack(images, axis=-1) 

def StackImages(img_index, mod0_dir, mod0_list, mod1_dir, mod1_list, mod2_dir, mod2_list, mod3_dir, mod3_list, dest_dir):
    img1_dir = os.path.join(mod0_dir, mod0_list[img_index])
    img2_dir = os.path.join(mod1_dir, mod1_list[img_index])
    img3_dir = os.path.join(mod2_dir, mod2_list[img_index])
    img4_dir = os.path.join(mod3_dir, mod3_list[img_index])

    img1 = cv2.imread(img1_dir, cv2.IMREAD_GRAYSCALE)  # Read as grayscale if needed
    img2 = cv2.imread(img2_dir, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(img3_dir, cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread(img4_dir, cv2.IMREAD_GRAYSCALE)

    images = [img1, img2, img3, img4]
    image = CombinedStack(images)

    # output_dir = os.path.join(dest_dir, mod0_list[img_index][:-8])
    output_dir = os.path.join(dest_dir, mod0_list[img_index])

    print(f"Saving image to...{output_dir}")
    # cv2.imwrite(f"{output_dir}.png", image)
    cv2.imwrite(f"{output_dir}", image)

def Main(): 
    mod_to_combine = []

    for mod in MODALITY:
        mod_to_combine.append(f"{IN_DIR}/{mod}_{DATASET}/images")

    split_list = ["test", "train", "val"]
    for split in split_list:
        mod0 = os.path.join(mod_to_combine[0], split)
        mod0_list = os.listdir(mod0)

        print(f"\nThis is the order {mod0}")

        mod1 = os.path.join(mod_to_combine[1], split)
        mod1_list = os.listdir(mod1)

        print(f"This is the order {mod1}")

        mod2 = os.path.join(mod_to_combine[2], split)
        mod2_list = os.listdir(mod2)

        print(f"This is the order {mod2}")

        mod3 = os.path.join(mod_to_combine[3], split)
        mod3_list = os.listdir(mod3)

        print(f"This is the order {mod3}")

        # Use ThreadPoolExecutor to sort the lists in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future0 = executor.submit(sorted, mod0_list)
            future1 = executor.submit(sorted, mod1_list)
            future2 = executor.submit(sorted, mod2_list)
            future3 = executor.submit(sorted, mod3_list)

            mod0_list = future0.result()
            mod1_list = future1.result()
            mod2_list = future2.result()
            mod3_list = future3.result()

        # check if they are sorted
        print(mod0_list[-5:-1])
        print(mod1_list[-5:-1])
        print(mod2_list[-5:-1])
        print(mod3_list[-5:-1])

        # create directory
        dest_dir_to_save = os.path.join(OUT_DIR, "images", split)
        CreateDir(dest_dir_to_save)

        # -------------------------------------------
        # LEAVE FOR DEBUGGING
        
        for img_index in range(len(mod0_list)):
             StackImages(img_index, mod0, mod0_list, 
                                 mod1, mod1_list, 
                                 mod2, mod2_list, 
                                 mod3, mod3_list, 
                                 dest_dir_to_save)
        # -------------------------------------------


        # with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        #      for img_index in range(len(mod0_list)):
        #          executor.submit(StackImages, img_index, mod0, mod0_list, 
        #                          mod1, mod1_list, 
        #                          mod2, mod2_list, 
        #                          mod3, mod3_list, 
        #                          dest_dir_to_save)

    print("All directories copied successfully.")

if __name__ == "__main__": 
    des="""
    This script is used to stack all four modalities into
    a single image
    """

    MODALITY = ["t1c" , "t1n", "t2f" ,"t2w"] 

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--dataset',type=str,help='options are: detection, segmentation, and yoloseg\t[None]')
    parser.add_argument('--modality', type=str, choices=MODALITY, nargs='+', help=f'BraTS dataset modalities to use\t[t1c, t1n, t2f, t2w]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    if args.in_dir is not None:
        IN_DIR = args.in_dir
    else: IN_DIR = "."
    if args.dataset is not None:
        DATASET = args.dataset
    else: DATASET = "yoloseg"
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    else: OUT_DIR = f"stacked_{DATASET}"
    if args.workers is not None:
        WORKERS = args.workers
    else: WORKERS = 10
    if args.modality is not None:
        MODALITY = [mod for mod in args.modality]

    Main()
    print("\nFinish stacking the dataset, please check your working directory")

