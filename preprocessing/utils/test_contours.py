import os
import cv2

"""
TEST IF MASK TO POLYGONS WORKS PROPERLY
"""

if __name__ == "__main__":
    label_dir = "t1c/labels/test"
    image_dir = "t1c/images/test"

    image_list = os.listdir(image_dir)

    for i, image in enumerate(image_list[296:319]):
        image_path = os.path.join(image_dir, image)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, image_binary_mask = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image_binary_mask_closed = cv2.morphologyEx(image_binary_mask, cv2.MORPH_CLOSE, kernel)
        image_binary_mask_open = cv2.morphologyEx(image_binary_mask_closed, cv2.MORPH_OPEN, kernel)

        print("This is image_path", image_path)

        # cv2.imshow("test", image)
        cv2.imshow("binarized", image_binary_mask)
        cv2.imshow("closed", image_binary_mask_closed)
        cv2.imshow("open", image_binary_mask_open)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #     label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # # Image processing
    # _, label_binary_mask = cv2.threshold(label, 250, 255, cv2.THRESH_BINARY)

    # contours, _ = cv2.findContours(image_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cleaned_image = image * image_binary_mask



