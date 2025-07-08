import cv2 as cv
import numpy as np
import os

if __name__ == "__main__":
    image_dir = os.path.join("t1c_cropped", "images", "test") 
    label_dir = os.path.join("t1c_cropped", "labels", "test") 

    image_list = os.listdir(image_dir)
    label_list = os.listdir(label_dir)

    image_list.sort()
    label_list.sort()

    key = 80

    image_path = os.path.join(image_dir, image_list[key])
    label_path = os.path.join(label_dir, label_list[key])

    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)

    print(np.unique(label))

    cv.imshow("image", image)
    cv.imshow("label", label)
    cv.waitKey(0)
    cv.destroyAllWindows()
