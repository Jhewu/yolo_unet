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
    """
    OFFICIAL CONVERTER, USE THIS INSTEAD
    """
    CreateDir(output_dir)
    convert_segment_masks_to_yolo_seg(masks_dir=dir_to_examine, output_dir=output_dir, classes=1)

# def MaskToPolygons(dir_to_examine, output_dir):
#     CreateDir(output_dir)

#     for mask_name in os.listdir(dir_to_examine):
#         mask_path = os.path.join(dir_to_examine, mask_name)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#         print(mask_path)
#         print(os.path.exists(mask_path))

#         # image processing
#         _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

#         H, W = mask.shape
#         contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # convert the contours to polygons
#         polygons = []
#         for cnt in contours:
#             # if cv2.contourArea(cnt) > 1:
#             polygon = []
#             for point in cnt:
#                 x, y = point[0]
#                 polygon.append(x / W)
#                 polygon.append(y / H)
#             polygons.append(polygon)

#         # check if polygons list is empty
#         if not polygons:
#             print(f"No polygons found for {mask_name}")

#         # ---------------------------------
#         # LEAVE THIS FOR TESTING
#         # cv2.imshow("test", binary_mask)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         # ---------------------------------

#         # write the polygons into a text file
#         output_file_path = '{}.txt'.format(os.path.join(output_dir, mask_name)[:-4])
#         print(f"Writing polygons to: {output_file_path}")
#         with open(output_file_path, 'w') as f:
#             for polygon in polygons:
#                 for p_, p in enumerate(polygon):
#                     if p_ == len(polygon) - 1:
#                         f.write('{}\n'.format(p))
#                     elif p_ == 0:
#                         f.write('0 {} '.format(p))
#                     else:
#                         f.write('{} '.format(p))
#             f.close()

def Main(): 
    for mod in MODALITY:
        label_dir = f"{IN_DIR}/{mod}_segmentation/labels"
        dest_label_dir = f"{OUT_DIR}/{mod}_yoloseg/labels"

        root = os.getcwd() 
        gt_dir = os.path.join(root, label_dir)

        # The list of splits (e.g., test, train and val)
        gt_dir_list = os.listdir(gt_dir)

        # -----------------------------------------------------------
        # LEAVE THIS FOR TESTING
        # input_dir = os.path.join(gt_dir, gt_dir_list[0])        
        # output_dir = os.path.join(dest_label_dir, gt_dir_list[0])
        # MaskToPolygons2(input_dir, output_dir)

        # -----------------------------------------------------------

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[0])        
            output_dir = os.path.join(dest_label_dir, gt_dir_list[0])
            executor.submit(MaskToPolygons2, input_dir, output_dir)

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[1])       
            output_dir = os.path.join(dest_label_dir, gt_dir_list[1]) 
            executor.submit(MaskToPolygons2, input_dir, output_dir)

        # mask to polygons on 1st directory
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            input_dir = os.path.join(gt_dir, gt_dir_list[2])   
            output_dir = os.path.join(dest_label_dir, gt_dir_list[2])     
            executor.submit(MaskToPolygons2, input_dir, output_dir)

if __name__ == "__main__": 
    # ------------------------------------------------------------------
    des="""
    Original script from computervisioneng on YouTube (slightly modified)

    An additional script that converts segmentation 
    masks (binary JPGs) into the YOLO segmentation polygons format
    """
    # ------------------------------------------------------------------

    MODALITY = ["t1c" , "t1n", "t2f" ,"t2w"] 

    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--in_dir", type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory prefix\t[None]')
    parser.add_argument('--modality', type=str, choices=MODALITY, nargs='+', help=f'BraTS dataset modalities to use\t[t1c, t1n, t2f, t2w]')
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
    if args.modality is not None:
        MODALITY = [mod for mod in args.modality]

    Main()
    print("\nFinish converting binary mask to polygon, check your directory for labels")