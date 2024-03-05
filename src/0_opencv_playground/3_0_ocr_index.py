import pytesseract as pt
from PIL import Image

image_file = "../data/index_02.jpg"
img = Image.open(image_file)

ocr_result = pt.image_to_string(img)

print(ocr_result)