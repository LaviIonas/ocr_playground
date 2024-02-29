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



```
    Local Feature Descriptors: Instead of simply flattening the images, consider using local feature descriptors such as Histogram of Oriented Gradients (HOG), Scale-Invariant Feature Transform (SIFT), or Local Binary Patterns (LBP). These descriptors capture local patterns and structures within the image, which can improve the discriminative power of the features.

    Filter Bank Features: Apply a filter bank to extract texture features from the images. Filters such as Gabor filters or Gaussian filters can capture texture information that may be useful for character recognition.

    Dimensionality Reduction: Use techniques like Principal Component Analysis (PCA) or Linear Discriminant Analysis (LDA) to reduce the dimensionality of the feature space while preserving important information. This can help improve computational efficiency and reduce the risk of overfitting.

    Normalization: Normalize the feature vectors to ensure that features are on the same scale. Common normalization techniques include Z-score normalization (subtracting the mean and dividing by the standard deviation) or Min-Max scaling (scaling features to a specified range).

    Augmentation: Apply data augmentation techniques such as rotation, translation, scaling, or adding noise to the images to increase the diversity of the training data and improve the robustness of the feature extraction process.

    Multi-Scale Features: Extract features at multiple scales to capture information at different levels of granularity. This can be achieved by resizing the images to different sizes or applying multi-scale feature extraction techniques.

    Domain-Specific Features: Consider incorporating domain-specific knowledge or heuristics into the feature extraction process. For OCR tasks, features such as stroke width, character height, or the presence of specific character components (e.g., serifs in letters) may be informative.

    Feature Selection: Perform feature selection to identify the most relevant features for character recognition. Techniques such as mutual information, chi-square test, or recursive feature elimination can help identify the most discriminative features.
```