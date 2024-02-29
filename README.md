# OCR Playground

# Section 1 : simple opencv image preprocessing 

### Workflow
```
PIL (Pillow)             => Open an Image
OpenCV                   => Change an Image
Tesseract (PyTesseract)  => OCR an Image
```

### Preprocessing Images for OCR
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

### Preprocessing for multiple columns of text
```
1. Blur image (to identify overall structure, and not focusing on text itself) 
2. Create threshold (and kernel) to separate text block 
3. Perform dilation (~white thickening)
4. Perform contour (finding boundaries)  
5. Perform loop to only draw boundrary box of specific size (to exclude small boxes)
```

# Section 2: simple OCR KNN from scratch (no libraries)

### Workflow
```
- No Libraries
- Use MNIST 28x28 bw hand drawn number data 
- Simple Euclidian distance between pixels and KNN to predict numbers
```

`Achieved Accuracy @ 1000 training samples predicting 200 numbers: 85.5%`

`Not bad, but so terrible inefficient, I don't even wanna bother waiting for larger samples`

# Section 3: simple OCR KNN but with efficient libraries (Numpy / sci-kit-learn)

### Workflow
```
- Use Numpy to read and process the images
- Use MNIST 28x28 bw hand drawn number data 
- Experiment with openCV to see if that changes OCR accuracy
- Try more efficient KNN algorithms (k-d trees, LSH)
```
