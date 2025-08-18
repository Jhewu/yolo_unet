import os
from PIL import Image

def count_black_images(directory_path):
    """
    Counts the number of images in a directory that are entirely black.

    Args:
        directory_path (str): The path to the directory containing the images.

    Returns:
        int: The total count of all-black images found.
    """
    # Check if the provided path is a valid directory
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a valid directory.")
        return 0

    # Counter for the number of all-black images
    all_black_images_count = 0
    # A list of common image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    print(f"Searching for all-black images in '{directory_path}'...")
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        
        # Check if the file is a regular file and has an image extension
        if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
            try:
                # Open the image file
                with Image.open(file_path) as img:
                    # Convert the image to grayscale ('L' mode) to simplify the check.
                    # An all-black image will have a maximum pixel value of 0 in grayscale.
                    grayscale_img = img.convert('L')
                    
                    # Use getextrema() to find the minimum and maximum pixel values
                    # This is much more efficient than iterating through every pixel.
                    min_val, max_val = grayscale_img.getextrema()
                    
                    if max_val == 0:
                        all_black_images_count += 1
                        print(f"  Found all-black image: {filename}")
            except Exception as e:
                # Catch potential errors when opening or processing a file
                print(f"  Could not process file {filename}: {e}")
    
    print("-" * 40)
    print(f"Finished searching. Total all-black images found: {all_black_images_count}")
    return all_black_images_count

if __name__ == "__main__":
    count_black_images("test")