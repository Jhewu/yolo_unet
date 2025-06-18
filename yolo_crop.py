import ultralytics
import os
from parameters import BEST_MODEL_DIR_PREDICT

IN_DIR = "datasets/data/images/test"
OUT_DIR = "cropped_images"
BATCH_SIZE = 32

def yolo_crop(): 
    return


def yolo_crop_cpu(): 
    return



def main(): 
    image_list = os.listdir(IN_DIR)

    # Construct the full directories of images
    image_full_paths = []
    [image_full_paths.append(os.path.join(IN_DIR, image)) for image in image_list]

    # [print(os.path.exists(image)) for image in image_full_paths]

    # Batch the directories
    print(len(image_full_paths))
    for i in range(0, len(image_full_paths), BATCH_SIZE): 
        print(i)
    







if __name__ == "__main__": 
    main()
