import os
import cv2

"""
TEST IF MASK TO POLYGONS WORKS PROPERLY
"""

if __name__ == "__main__":
    label_dir = "t1c_yoloseg/labels/test"
    image_dir = "t1c_segmentation/labels/test"

    image_list = os.listdir(image_dir)

    for i, image in enumerate(image_list[190:205]):
        image_path = os.path.join(image_dir, f"{image}.png")
        label_path = os.path.join(label_dir, f"{image}.txt")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        print("This is image_path", image_path)
        print("This is label_path", label_path)

        with open(label_path, 'r') as f:
            print(f.read())
        f.close()

        cv2.imshow("test", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


