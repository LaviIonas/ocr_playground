import pytesseract 
from PIL import Image

img_file = "../data/page_01.jpg"
no_noise = "temp/no_noise.jpg"

img = Image.open(no_noise)

ocr_result = pytesseract.image_to_string(no_noise)

print(ocr_result)

