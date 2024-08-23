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

This lab is centered around introducing basic concepts of data analysis and statistics, and applying those concepts to real-world data, specifically housing data. Let me break it down step by step:

### **Lab Structure Overview**

1. **Task 1: Create Basic Functions**
   - You're asked to build several basic statistical functions **from scratch**—no using Python's built-in functions for these.
   - These functions include:
     - **Min**: Find the smallest value.
     - **Max**: Find the largest value.
     - **Mean**: Calculate the average.
     - **Variance**: Measure the spread of data points around the mean.
     - **Standard Deviation**: Another measure of spread, based on variance.
     - **Median**: The middle value in a sorted dataset.
     - **Median Absolute Deviation (MAD)**: The median of the absolute differences from the median.

   **Why?** The goal is to understand how these fundamental statistical measures are calculated without relying on shortcuts. It builds a deeper intuition for what each measure represents.

2. **Task 2: Get Insights from a List of Grades**
   - Here, you're simulating the role of a teacher analyzing student grades. After defining a list of grades, you apply the functions you wrote in Task 1 to find the min, max, mean, median, standard deviation, and MAD.
   - **Insight**: The metrics you compute help describe how the students performed:
     - **Min/Max**: Show the range of grades.
     - **Mean**: Represents the average score.
     - **Standard Deviation**: Tells you how varied the scores are.
     - **Median**: Shows the middle score, which can be useful if the data is skewed (e.g., if there are outliers).
     - **MAD**: Measures how much variation there is from the median, giving you a sense of consistency around the middle.

3. **Task 3: Load and Inspect the Housing Data**
   - This task involves working with real-world data in the form of a CSV file. You're tasked with loading the housing data using **Pandas**, a Python library for data manipulation.
   - The goal is to:
     - **Load the data**: This is important because it’s what you will work with in Task 4.
     - **Inspect the data**: Get an idea of what’s in the dataset by printing the first few rows and some basic info like the number of rows, columns, and data types.

4. **Task 4: Apply Functions to the Housing Dataset**
   - Now that the data is loaded, you'll apply your statistical functions (or slightly modify them) to the housing dataset to extract useful information.
     1. **Count the Number of Districts**: Essentially, count the rows in the dataset (which represent districts).
     2. **Calculate the Mean of House Values**: Use your `mean()` function to compute the average house value.
     3. **Create Histograms**: Histograms allow you to visualize the distribution of different variables like the number of households, median income, housing age, and median house value.
     4. **Analyze the Histograms**: Look at patterns in the graphs, especially at the tails of distributions (where the values get very high or very low). You are asked to notice trends, such as whether the median house value distribution is skewed or if there are unusual patterns in housing age.
     5. **Discuss the Magnitude of Values in Median House Value**: Think about how the house values might have been processed. For example, if the values are unusually large, maybe they are expressed in thousands or another unit.

   - **Additional Task**: Group the data by ocean proximity (a column in the dataset) and calculate the average house value for each proximity category (e.g., "near ocean", "inland").

5. **Final Analysis Questions**:
   - **Student Grades**: Which metric (mean, median, standard deviation, etc.) is best to analyze student performance? Typically, the **mean** is a good overall indicator, but **median** can be helpful if there are outliers (very low or very high grades).
   - **Income Distribution**: For income analysis, the **median** is often better than the mean because it is less affected by extremely high incomes (like those of billionaires). The **mean** could be skewed by those outliers, giving a false sense of what a typical income looks like.

---

### **What You’re Learning in This Lab**

- **Basic Statistics**: You’re learning to calculate key statistics (mean, median, etc.) without relying on built-in functions, which helps you understand how these measures work under the hood.
- **Data Handling**: You’re introduced to **Pandas**, a powerful tool for working with datasets in Python. This will be valuable in handling real-world datasets later on.
- **Data Visualization**: Creating histograms lets you visualize how data is distributed, helping you make sense of patterns and outliers.
- **Real-World Application**: Working with the housing dataset introduces you to common tasks in data analysis, such as calculating averages for different groups (ocean proximity), and examining distributions of key features like income or house value.

---

### **Key Takeaways**

1. **Statistical Functions**: Learn how to calculate important metrics like mean, variance, and median from scratch.
2. **Data Insights**: Understand what each statistic means and how it can help analyze data.
3. **Pandas**: Use this Python library to load, inspect, and manipulate datasets.
4. **Visualization**: Plot histograms to identify trends and outliers in data.
5. **Real Data Application**: Apply your custom functions to a real dataset, practicing skills that are commonly used in data analysis.

---

## Lab 2: Linear and Polynomial Regression

This lab aims to teach the process of performing linear and polynomial regression on datasets. The goal is to understand the concepts of regression analysis and how to implement and evaluate regression models, both manually and using machine learning libraries such as Scikit-learn.

Let’s break down each section of the lab step by step:

---

### **Task 1: Loading a Subset of Data**

The first task involves loading a dataset (`inc_subset.csv`) that contains income data based on age. The goal is to write a custom CSV loader using basic Python file I/O (no Pandas or CSV libraries), which is essential for understanding how data is parsed and read into your program.

#### Code Breakdown:

- **Custom CSV Loader**: The `load_csv` function reads the file line by line, skips the header, and splits the data by commas.
- **Data Structure**: The data is loaded into a list of lists, where each sublist represents a row from the CSV.

The function ensures that the values from the CSV file are properly loaded and converted into numeric data. It’s also important to inspect the data using print statements or a debugger to verify that it is loaded correctly.

**Key Learning**: This task reinforces how basic file handling works and the importance of verifying the integrity of the data before performing any operations.

---

### **Task 2: Linear Regression on a Data Subset**

#### 2.1 Perform Linear Regression
Linear regression is performed on the subset of the dataset (age vs. income). The model is implemented from scratch using basic mathematical formulas for linear regression.

- **Training and Validation Split**: The dataset is split into training (80%) and validation (20%) subsets. Splitting the data ensures that you evaluate your model on unseen data to check how well it generalizes.
- **Formula for Linear Regression**:
  - **Slope (m)** and **Intercept (b)** are calculated using the least-squares method:
    - \( m = \frac{\sum (X - \text{mean}(X)) (Y - \text{mean}(Y))}{\sum (X - \text{mean}(X))^2} \)
    - \( b = \text{mean}(Y) - m \times \text{mean}(X) \)

#### 2.2 Scatter Plot and Regression Line
A scatter plot is generated with age on the X-axis and income on the Y-axis, and the regression line is plotted over the data.

#### 2.3 Predict Values for Validation Data
Using the regression model (`y = m * X + b`), predictions are made for the validation dataset (ages from the validation data).

#### 2.4 Evaluate the Model with MSE
Mean Squared Error (MSE) is computed to evaluate the model’s accuracy. MSE calculates the average of the squared differences between the actual and predicted values:
- \( \text{MSE} = \frac{1}{n} \sum (Y_{\text{true}} - Y_{\text{predicted}})^2 \)

A lower MSE indicates a better model fit.

---

### **Task 3: Linear Regression on the Full Dataset**

#### 3.1 Load and Clean the Full Dataset
Here, you load a larger dataset (`inc_utf.csv`) that contains income data by age and region. Some data cleaning is required (e.g., converting "100+" to numeric and handling non-numeric values).

- **Data Grouping**: The data is grouped by **age** using Pandas, and the mean income across all regions is calculated.

#### 3.2 Perform Linear Regression on Grouped Data
Just like in Task 2, linear regression is applied to the grouped data (age vs. income). The grouped data has more age groups, making the analysis more comprehensive.

#### 3.3 Create Scatter Plot and Regression Line
The scatter plot and regression line are visualized for the full dataset, similar to Task 2.

#### 3.4 Predict and Evaluate the Model
Predictions are made for the validation data, and the MSE is calculated again to evaluate the model's performance. This allows you to compare the model’s accuracy on the full dataset versus the subset dataset.

---

### **Task 4: Reflections**

In this task, you reflect on the differences between the two linear regressions:
- **Graphical Comparison**: You observe differences in the regression lines generated by the subset vs. full dataset. The full dataset likely provides a more accurate representation due to having more data points.
- **MSE Comparison**: Compare the MSE values between the two models. A lower MSE indicates a better fit, and you may find that the full dataset results in a lower MSE because more data leads to better predictions.

---

### **Task 5: Polynomial Regression with Hyperparameter Tuning**

#### 5.1 Polynomial Regression
In polynomial regression, you extend the linear model to include higher-order polynomial features. This allows the regression line to fit non-linear data patterns. Scikit-learn’s `PolynomialFeatures` class is used to transform the input features into polynomial features.

#### 5.2 Hyperparameter Tuning
The degree of the polynomial is a **hyperparameter**. You test different polynomial degrees (e.g., 2, 3, 5, and 8) to see which one fits the data best.

#### 5.3 Evaluating with MSE
For each polynomial degree, you compute the MSE to evaluate how well the model fits the data. The model with the lowest MSE is considered the best fit.

#### 5.4 Graphical Comparison
You plot the polynomial regression line with the optimal degree alongside the original linear regression line. This comparison visually shows how well the polynomial regression model fits the data.

---

### **Key Takeaways**

1. **Linear Regression**: 
   - The goal is to find the best-fit line that describes the relationship between two variables (age and income).
   - You manually implement the linear regression formula and evaluate the model’s performance using MSE.

2. **Data Splitting**: 
   - Splitting data into training and validation sets is essential for evaluating model performance on unseen data.
   
3. **Polynomial Regression**:
   - Polynomial regression can model more complex, non-linear relationships between variables by adding higher-order terms (e.g., \(x^2\), \(x^3\)).
   - Hyperparameter tuning helps in finding the best polynomial degree to avoid underfitting or overfitting the data.

4. **Evaluation with MSE**: 
   - MSE is a widely used metric to evaluate regression models. It provides insight into how far the predicted values are from the actual values.
   
5. **Reflections**:
   - By comparing the linear and polynomial regression models, you gain insights into which approach is more effective for the given dataset.

---

### Graphical Examples
The graphs you generate (scatter plots with regression lines) provide a visual way to understand how well the model fits the data. The graph also shows how polynomial regression may perform better than linear regression when the relationship between variables is non-linear.

---

## Lab 3: Classification with K-means Clustering and Hyper-parameter Optimization using Grid Search

This lab is all about **K-means clustering** and **hyper-parameter optimization** using **grid search**. The goal is to:
- Understand the K-means clustering algorithm by implementing it from scratch.
- Cluster a dataset of **rent** and **income** values using K-means.
- Use **grid search** to find the optimal number of clusters (k) by maximizing the **silhouette score**.
- Classify new data points based on the clustering model.

Here’s a breakdown of the tasks and what each section of the code is doing:

---

### **Task 1: Classification with K-means**

#### **Task 1.1: Load the Dataset**
- The dataset `rent_vs_inc.csv` contains information about regions’ **annual rent** and **average yearly income**.
- In the `load_csv` function, you're loading this dataset in a similar way as in previous labs. The first few rows are printed to inspect and verify that the data is loaded correctly.

**Key Concepts:**
- **CSV File Loading**: Custom loading of CSV without libraries like pandas (only using basic file I/O).
  
#### **Task 1.2: Scatter Plot**
- After loading the data, you visualize the relationship between **rent** and **income** by creating a **scatter plot** using **matplotlib**.

**Key Concepts:**
- **Visualization**: Scatter plots help identify patterns or clusters in the data visually.
- The X-axis is the **rent**, and the Y-axis is the **income**.

---

#### **Task 1.3: Implement K-means Clustering from Scratch**
- K-means clustering is an unsupervised machine learning algorithm that divides the data into **k clusters** based on feature similarity (in this case, rent and income).
  
**Steps in K-means:**
1. **Initialize Centroids**: Randomly select `k` points from the dataset as the initial centroids (starting points for the clusters).
2. **Assign Points to Nearest Centroid**: For each point in the dataset, compute its distance to each centroid and assign it to the closest centroid.
3. **Recompute Centroids**: For each cluster, compute the mean of the points assigned to that cluster. This mean becomes the new centroid.
4. **Repeat**: Repeat the assignment and recomputation steps until the centroids stop changing significantly (convergence) or until a set number of iterations (e.g., 10 iterations).
  
**Key Functions:**
- **`euclidean_distance()`**: Computes the Euclidean distance between two points. This is used to find the closest centroid for each data point.
- **`kmeans()`**: Implements the K-means clustering algorithm. It returns the final centroids and the clusters after a fixed number of iterations.

#### **Visualizing Clusters**
After performing K-means clustering, you plot the data points, with each point colored based on the cluster it belongs to. You also plot the centroids (cluster centers) with a different marker.

---

### **Task 2: Hyper-parameter Optimization**

#### **Task 2.1: Silhouette Score and Grid Search**

In K-means, you need to choose the **number of clusters (k)**. If k is too small or too large, the clusters may not represent the data well. The **silhouette score** is a metric that helps determine how well-defined your clusters are.

- **Silhouette Score (S(i))**:
  - \( a(i) \) is the **average distance** between a point and all other points in its own cluster.
  - \( b(i) \) is the **average distance** between a point and all points in the nearest other cluster.
  - The silhouette score for a point \( i \) is calculated as:
    \[
    S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
    \]
  - The silhouette score ranges from -1 to 1:
    - **1**: Perfect clustering.
    - **0**: Points are on the border of two clusters.
    - **-1**: Points are assigned to the wrong cluster.
  
**Grid Search**:
- **Grid search** is a method used to find the optimal hyperparameters by trying different values and measuring performance (in this case, the silhouette score).
- In this task, you try different values of **k** (from 2 to 10) and calculate the **silhouette score** for each value of **k**. The optimal value of **k** is the one that maximizes the silhouette score.

#### **Code Explanation**:
- **`silhouette_coefficient()`**: Computes the silhouette score for a clustering.
- **`grid_search()`**: Iterates over different values of k (number of clusters) and calculates the silhouette score for each. This helps identify the optimal number of clusters.

#### **Visualizing Silhouette Scores**:
- You plot the silhouette scores for each k to see which k maximizes the score. The peak of this plot corresponds to the optimal number of clusters.

---

#### **Task 2.2: Scatter Plot with Optimal Number of Clusters**
- After finding the optimal number of clusters (k), you run K-means again with this value of k.
- You create a new scatter plot, where each cluster is assigned a different color, and the centroids are also plotted.

---

#### **Task 2.3: Classify New Data Points**
- Now, you classify three new data points (new regions with known rent and income) based on the trained K-means model.
- For each new data point, you calculate its distance to each cluster’s centroid and assign it to the closest centroid (cluster).
  
**Key Concepts**:
- **Classification** in K-means is done by assigning a new point to the nearest cluster centroid.

---

### **Optional Task: N-dimensional Grid Search Optimizer**
- In this optional task, you could expand the grid search algorithm to optimize **multiple hyperparameters** (e.g., number of clusters and another parameter).
- This requires iterating over all combinations of hyperparameters and calculating an evaluation metric (such as silhouette score) for each combination.

---

### **Summary of Key Learnings**

1. **K-means Clustering**:
   - You learned how to implement the K-means algorithm from scratch.
   - K-means groups similar data points into clusters, where each point is assigned to the cluster with the closest centroid.

2. **Visualization**:
   - Scatter plots are used to visualize the relationship between the data (rent vs. income).
   - Clusters are visualized with different colors, and the centroids are highlighted to show where the "centers" of the clusters are.

3. **Hyper-parameter Tuning**:
   - **Grid search** is used to optimize the number of clusters (k) by evaluating different values of k with the **silhouette score**.
   - The silhouette score helps measure how well-defined the clusters are.

4. **New Data Classification**:
   - The trained K-means model is used to classify new data points by assigning them to the nearest centroid.

5. **Silhouette Score**:
   - A key evaluation metric in clustering tasks. It provides a way to assess the quality of clustering and helps in choosing the optimal number of clusters.

---

### **Conclusion**

This lab gives you practical experience with **K-means clustering**, **hyper-parameter tuning** using **grid search**, and clustering evaluation using the **silhouette score**. You also learned how to classify new data points based on the K-means model. This is foundational knowledge for understanding unsupervised learning techniques and how they can be applied to real-world data.

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
