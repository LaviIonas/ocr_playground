# OCR Playground

# Section 1 : simple opencv image preprocessing 

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

# Section 2: simple OCR KNN from scratch (no libraries)

## Workflow
```
- No Libraries
- Use MNIST 28x28 bw hand drawn number data 
- Simple Euclidian distance between pixels and KNN to predict numbers
```

`Achieved Accuracy @ 1000 training samples predicting 200 numbers: 85.5%`

`Not bad, but so terrible inefficient, I don't even wanna bother waiting for larger samples`

# Section 3: numpy optimized OCR simple KNN, alternative feature extraction

## Workflow
```
- Use Numpy to read and process the images faster
- Same Data as before
- Same KNN function but optimized with numpy
- Try different feature extraction methods
- Try different KNN variations 

Note: Accuracy being tested at 10,000 samples for 200 guesses where k is 3
No hyperparameter optimization is done yet, save that for later as a treat
```

## Alternative Extraction Methods
`Insead of just flattening the feature vector, we can try other methods`

`Yeilds: 45%`

### Local Binary Patterns (LBP)
`Local pattern in the image, compare intensity of pixels given its surrounding pixels`

`Yields: 90%`

### Histogram of Oriented Gradients (HOG)
`Captures distribution of gradients in localized regions, useful to detect edges and shapes`

`Yields: 91%`

## Alternative KNN fucntions
*using hog at k=3*

### Weighted KNN
`Instead of most commonly occuring neightbor, weighted average of labels of neighbors. Closer neightbors contribute more than ones farther away.`

`Yields: 89.5%`

### Radius-Based KNN
`Instead of a fixed number of neightbors, we consider all neighbors in a radius around the sample`

`Yields: 86.5%`

### KD-Tree
`partition high dimnetional data into trees to reduce computation time.`

`Yields: 89.5%`

# Section 4: simple CNN from scratch, without OCR 



