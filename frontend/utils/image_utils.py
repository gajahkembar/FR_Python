from PIL import Image
import io

def load_image_bytes(image_file):
    image = Image.open(image_file)
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    return byte_arr.getvalue()