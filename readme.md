# Machine Learning Labs: Overview

This repository provides a summary of the labs covered in the course, which focus on building foundational knowledge in Python programming, data handling, regression analysis, clustering, and image processing. Below is a concise breakdown of each lab and its main tasks.

## Lab 0: Getting Started with Programming in Python

### Objective:
- Familiarize yourself with the PyCharm IDE and set up required software and libraries.

### Tasks:
1. **Install software**: Set up PyCharm, install necessary libraries (Numpy, Pandas, etc.).
2. **Create a Python script**: Write basic Python functions like `print` and simple functions.
3. **Linear model example**: Load a provided test file to preview upcoming course material.

---

## Lab 1: Data and Statistics

### Objective:
- Learn how to calculate basic statistics and perform simple data analysis on a dataset.

### Tasks:
1. **Basic statistical functions**: Implement functions to calculate min, max, mean, variance, standard deviation, median, and median absolute deviation without using built-in Python functions.
2. **Data analysis**: Apply the functions to a list of student grades.
3. **Load and inspect housing data**: Load the `housing.csv` file and explore the dataset.
4. **Apply functions to real dataset**: Use the statistical functions to analyze the housing dataset and create visualizations.

---

## Lab 2: Linear and Polynomial Regression

### Objective:
- Perform regression analysis on a dataset and evaluate model performance.

### Tasks:
1. **Load data**: Load a subset of data (`inc_subset.csv`) without using Pandas.
2. **Linear regression**: Implement linear regression from scratch and split the data into training and validation sets.
3. **Full dataset regression**: Perform linear regression on a larger dataset (`inc_utf.csv`), group data by age, and plot results.
4. **Polynomial regression**: Use Scikit-learn’s PolynomialFeatures to implement polynomial regression and optimize using hyperparameter tuning.

---

## Lab 3: K-means Clustering and Hyper-parameter Optimization

### Objective:
- Implement K-means clustering from scratch and optimize the number of clusters using grid search.

### Tasks:
1. **K-means implementation**: Manually implement K-means and apply it to the `rent_vs_inc.csv` dataset to classify rent vs income.
2. **Hyper-parameter optimization**: Use grid search and silhouette scores to find the optimal number of clusters.
3. **Classify new data**: Use the trained K-means model to predict clusters for new data points.

---

## Lab 4: Image Processing with Scikit-Image

### Objective:
- Learn basic image processing techniques using the Scikit-Image library.

### Tasks:
1. **Load and visualize images**: Load images like `coins.jpg` and `astronaut.jpg` and inspect their properties.
2. **Color space conversion**: Convert color images to grayscale.
3. **Rescale and resize images**: Modify the dimensions of an image using `rescale` and `resize` functions.
4. **Image thresholding**: Segment images using basic thresholding techniques.
5. **Template matching**: Implement template matching to locate objects within an image.

---

### Summary:
Each lab builds upon key programming and data analysis concepts, moving from basic Python operations in Lab 0 to advanced topics like regression, clustering, and image processing in Labs 1 through 4. By the end of these labs, you'll have a strong foundation in handling real-world data and applying machine learning techniques.
