import os
from PIL import Image
import numpy as np

def count_masks_with_values(directory_path, threshold=10):
    """
    Examines a directory of grayscale mask images and counts how many have
    pixels with values greater than or equal to a specified threshold.

    Args:
        directory_path (str): The path to the directory containing the mask images.
        threshold (int): The minimum pixel value to be considered 'present'.
                         Defaults to 10 as requested.

    Returns:
        tuple: A tuple containing two integers:
               - The count of images with pixel values >= threshold.
               - The count of images without any pixel values >= threshold.
    """
    # Initialize counters for the two classes
    with_values_count = 0
    without_values_count = 0

    # Get a list of all files in the specified directory
    files = os.listdir(directory_path)

    print(f"Scanning directory: {directory_path}")
    print(f"Using a pixel value threshold of: {threshold}")
    print("-" * 40)

    # Iterate through each file in the directory
    for filename in files:
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)

        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            try:
                # Open the image using Pillow (PIL)
                img = Image.open(file_path)

                # Convert the image to a NumPy array for fast pixel value checking
                # The 'L' mode ensures it's treated as a single-channel grayscale image
                img_array = np.array(img.convert('L'))

                # Check if any pixel value is greater than or equal to the threshold
                # The .any() method is highly efficient for this check
                if np.any(img_array >= threshold):
                    with_values_count += 1
                else:
                    without_values_count += 1

            except IOError:
                # Skip files that are not valid images
                print(f"Skipping non-image file: {filename}")
                continue
    
    # Calculate total and percentages for reporting the imbalance
    total_images = with_values_count + without_values_count
    
    # Avoid division by zero
    if total_images == 0:
        print("No images found in the directory.")
        return 0, 0

    with_values_percentage = (with_values_count / total_images) * 100
    without_values_percentage = (without_values_count / total_images) * 100

    print("\n--- Imbalance Report ---")
    print(f"Images with pixel values (>= {threshold}): {with_values_count} ({with_values_percentage:.2f}%)")
    print(f"Images with no significant pixel values: {without_values_count} ({without_values_percentage:.2f}%)")
    print(f"Total images scanned: {total_images}")
    print("-" * 40)

    return with_values_count, without_values_count

# Example usage of the function
if __name__ == "__main__":
    # IMPORTANT: Change this path to the directory you want to analyze
    # For this example, we assume a directory named 'masks' exists in the same
    # location as the script.
    directory_to_check = "val"

    # You can call the function with a different threshold if you like
    # For example, to check for any non-zero pixels:
    # count_masks_with_values(directory_to_check, threshold=1)
    
    # Or to use the default threshold of 10:
    count_masks_with_values(directory_to_check)
