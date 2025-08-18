import piexif
import cv2
from PIL import Image

def write_metadata(image_path, center_x, center_y, width, height): 
    img = Image.open(image_path)
    exif = img.getexif()
    exif_bytes = exif.tobytes()

    exif_dict = piexif.load(exif_bytes)

    # Convert the custom metadata to a format that can be written in EXIF
    new_data = {
        piexif.ExifIFD.UserComment: f"{center_x},{center_y},{width},{height}".encode('utf-8')
    }
    
    exif_dict["Exif"].update(new_data)
    
    # Create the bytes for writing to the image
    exif_bytes = piexif.dump(exif_dict)
    
    # Save the image with the new EXIF data
    img.save("chicken.png", exif=exif_bytes)

def read_metadata(image_path):
    img = Image.open(image_path)
    exif = img.getexif()
    exif_bytes = exif.tobytes()
    exif_dict= piexif.load(exif_bytes)

    # Grab the raw bytes of the UserComment tag
    raw_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)

    if raw_comment is None:
        return None

    # Convert the tuple (or bytes) to a real string
    # The EXIF spec says the first 8 bytes are an encoding prefix.
    # If you wrote the string yourself (without a prefix) it will
    # simply be the raw UTF‑8 bytes, so we can decode directly.
    comment = bytes(raw_comment).decode("utf-8", errors="ignore")
    return comment

if __name__ == "__main__": 
    # image_path = "modified_image.png"
    image_path = "BraTS-SSA-00046-00057-t1c.png"

    print(read_metadata(image_path))

    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # pil_image = Image.fromarray(image)

    # print(pil_image)

    # exif = pil_image.getexif()
    # exif_bytes = exif.tobytes()

    # exif_dict = piexif.load(exif_bytes)

    # print(exif_dict)

