import numpy as np
import matplotlib.pyplot as plt
import cv2


def crop_center(image, x_center, y_center, crop_size):
    """
    Crops an image around a center point, padding with zeros if necessary.
    
    Args:
        image: Input image (numpy array)
        x_center, y_center: Center coordinates for the crop
        crop_size: Size of the output crop (assumes square crop)
    
    Returns:
        Cropped image of size (crop_size, crop_size)
    """
    height, width = image.shape[0], image.shape[1]
    half_crop = crop_size // 2
    
    # Calculate desired crop boundaries
    x1 = x_center - half_crop
    y1 = y_center - half_crop
    x2 = x1 + crop_size  # Ensure exact crop_size
    y2 = y1 + crop_size
    
    # Calculate actual crop boundaries (clipped to image)
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(width, x2)
    y2_clip = min(height, y2)
    
    # Extract the portion of image within bounds
    cropped = image[y1_clip:y2_clip, x1_clip:x2_clip]
    
    # Calculate padding needed
    pad_left = x1_clip - x1
    pad_top = y1_clip - y1  
    pad_right = x2 - x2_clip
    pad_bottom = y2 - y2_clip
    
    # Apply padding if necessary
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        if len(image.shape) == 3:  # Color image
            cropped = np.pad(cropped, 
                           ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                           mode='constant', constant_values=0)
        else:  # Grayscale image
            cropped = np.pad(cropped, 
                           ((pad_top, pad_bottom), (pad_left, pad_right)), 
                           mode='constant', constant_values=0)
    
    return cropped

def draw_square_opencv(image, x_center, y_center, square_size, thickness=1, color=255):
    """
    Alternative implementation using OpenCV's rectangle function.
    Simpler but requires OpenCV installation.
    """
    result_image = image.copy()
    half_size = square_size // 2
    
    pt1 = (x_center - half_size, y_center - half_size)
    pt2 = (x_center + half_size, y_center + half_size)
    
    if len(image.shape) == 3:
        if isinstance(color, (int, float)):
            draw_color = (color, color, color)
        else:
            draw_color = color
    else:
        draw_color = color
    
    # Use -1 for filled rectangle, positive value for border thickness
    cv_thickness = -1 if thickness >= square_size // 2 else thickness
    
    cv2.rectangle(result_image, pt1, pt2, draw_color, cv_thickness)
    return result_image

if __name__ == "__main__": 
    image = "BraTS-SSA-00041-00046-t1c.png"
    label = "BraTS-SSA-00041-00046-t1c copy.png"

    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label, cv2.IMREAD_UNCHANGED)

    from custom_predictor.custom_detection_predictor import CustomDetectionPredictor
    CONFIDENCE = 0.7
    MODEL_DIR = "yolo_weights/best.pt"

    args = dict(conf=CONFIDENCE)  
    predictor = CustomDetectionPredictor(overrides=args)
    predictor.setup_model(MODEL_DIR)

    image_results = predictor(image)

    for result in image_results: 
        boxes = result.boxes
        if len(boxes) > 0: 
            coords = boxes.xywh[0]
            center_x = int(coords[0])
            center_y = int(coords[1])

    crop = crop_center(image, center_x, center_y, 64)
    crop_label = crop_center(label, center_x, center_y, 64)

    # crop = crop_center(image, center_x, center_y, 64)
    draw = draw_square_opencv(image, center_x, center_y, 64)
    draw_label = draw_square_opencv(label, center_x, center_y, 64)

    plt.imshow(image)
    plt.show()

    plt.imshow(label)
    plt.show()

    plt.imshow(draw)
    plt.show()

    plt.imshow(draw_label)
    plt.show()

    plt.imshow(crop)
    plt.show()

    plt.imshow(crop_label)
    plt.show()

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()