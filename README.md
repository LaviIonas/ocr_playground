# OCR Playground

## Workflow
```
PIL (Pillow)             => Open an Image
OpenCV                   => Change an Image
Tesseract (PyTesseract)  => OCR an Image
```

## Preprocessing Images for OCR
```
1. Inverted Images
2. Rescaling
3. Binarization
4. Noise Removal
5. Dilation and Erosion
6. Rotation / Deskewing 
7. Removing Borders
8. Missing Borders
9. Transparency / Alpha Channel
```

## Preprocessing for multiple columns of text
```
1. Blur image (to identify overall structure, and not focusing on text itself) 
2. Create threshold (and kernel) to separate text block 
3. Perform dilation (~white thickening)
4. Perform contour (finding boundaries)  
5. Perform loop to only draw boundrary box of specific size (to exclude small boxes)
```