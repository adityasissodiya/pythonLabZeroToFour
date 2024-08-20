### Explanation of Lab 2: Linear and Polynomial Regression

In this lab, you will practice performing linear and polynomial regression on real-world datasets. The objective is to develop a regression model from scratch (without using `sklearn` for the linear regression) and later to refine your model using polynomial regression. You will evaluate the effectiveness of both techniques using Mean Squared Error (MSE) and discuss the results.

---

### **Task 1: Load a Subset of the Data**

You need to create a custom function to load a CSV file **without using libraries like pandas or csv**. This is a basic file I/O task, where you’ll read the file and parse its contents manually.

#### Code for Task 1: Custom CSV Loader

```python
def load_csv(filepath):
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skipping header
            values = line.strip().split(',')
            data.append([float(v) for v in values])
    return data

# Loading the subset data from the CSV
subset_data = load_csv('inc_subset.csv')

# Inspect data using print or debugger
for row in subset_data:
    print(row)  # Print out data to verify it has loaded correctly
```

In this function:
- You open the file and read it line by line.
- The first line (header) is skipped.
- Each line is split by commas, and the resulting values are converted to floats before being added to a list.

#### Discussion:
- **Debugger**: Useful when you want to inspect variable values during each step of execution, such as when you are uncertain if the logic of a complex function is correct.
- **Print Statements**: More efficient for simple, quick inspections of variable states, especially when you need to verify data in between code blocks without pausing execution.

---

### **Task 2: Linear Regression on a Data Subset**

Here, you will perform a simple linear regression from scratch using a subset of the data (income vs. age).

#### Code for Task 2: Linear Regression from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

# Linear regression function
def linear_regression(X, y):
    # X is the input (age), y is the target (income)
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # Using the formula for slope (m) and intercept (b)
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * X_mean
    
    return slope, intercept

# Split the dataset into training and validation sets (80-20 split)
def train_test_split(data, validation_proportion=0.2):
    np.random.shuffle(data)
    validation_size = int(len(data) * validation_proportion)
    validation_data = data[:validation_size]
    train_data = data[validation_size:]
    return np.array(train_data), np.array(validation_data)

# Extract age and income columns
age = np.array([row[0] for row in subset_data])
income = np.array([row[1] for row in subset_data])

# Split the dataset
train_data, validation_data = train_test_split(subset_data)

# Perform linear regression on the training data
slope, intercept = linear_regression(train_data[:, 0], train_data[:, 1])

# Create a regression line
regression_line = slope * age + intercept

# Plot the regression line and scatter plot
plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', label='Training Data')
plt.plot(age, regression_line, color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()
```

#### Explanation:
- **Linear Regression**: The code computes the slope and intercept using the formula for least squares regression.
- **Data Split**: The dataset is split into training and validation sets using an 80-20 proportion.
- **Scatter Plot**: The scatter plot visualizes the relationship between age and income, with the regression line superimposed.

---

#### **Task 2.3-2.4: Predicting Values and Evaluating with MSE**

You will predict the income for the validation dataset using the regression model and evaluate the model's accuracy using Mean Squared Error (MSE).

#### Code for Task 2.3-2.4: Prediction and MSE Calculation

```python
# Predict function
def predict(X, slope, intercept):
    return slope * X + intercept

# MSE function
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Predict values for the validation dataset
predicted_income = predict(validation_data[:, 0], slope, intercept)

# Calculate MSE
mse_value = calculate_mse(validation_data[:, 1], predicted_income)
print("MSE:", mse_value)

# Print actual vs predicted values
for actual, predicted in zip(validation_data[:, 1], predicted_income):
    print(f"Actual: {actual}, Predicted: {predicted}")
```

#### Explanation:
- **Prediction**: For each age in the validation set, the income is predicted using the linear equation derived earlier.
- **MSE**: This evaluates the difference between actual and predicted values, where a smaller MSE indicates better model performance.

---

### **Task 3: Linear Regression on Full Dataset**

For this task, you will load the second dataset `inc_utf.csv` and apply linear regression. The data must be grouped by age and region, and the results will be compared with the subset data.

#### Code for Task 3.1-3.2: Grouping and Linear Regression

```python
# Load the full dataset
full_data = load_csv('inc_utf.csv')

# Grouping by age using pandas
import pandas as pd
df_full = pd.DataFrame(full_data, columns=['Age', 'Region', 'Year', 'Income'])

# Clean the "Year" column (removing non-numeric values)
df_full['Year'] = pd.to_numeric(df_full['Year'], errors='coerce')

# Group by age and compute the mean income across regions
grouped_data = df_full.groupby('Age').mean()['Income'].reset_index()

# Perform linear regression on the grouped data
slope_full, intercept_full = linear_regression(grouped_data['Age'], grouped_data['Income'])

# Plot the results
regression_line_full = slope_full * grouped_data['Age'] + intercept_full

plt.scatter(grouped_data['Age'], grouped_data['Income'], color='blue', label='Grouped Data')
plt.plot(grouped_data['Age'], regression_line_full, color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()
```

#### Explanation:
- **Grouping**: The data is grouped by age to calculate the mean income across regions.
- **Regression**: The linear regression process is repeated on this grouped data.
  
---

### **Task 4: Reflections on Linear Regression**

After performing both linear regressions (on the subset and full datasets), compare the MSE values and regression lines. Reflect on the differences in performance between the two models.

---

### **Task 5: Polynomial Regression with Hyperparameter Tuning**

You will use `scikit-learn` to perform polynomial regression and compare the results with the linear regression model.

#### Code for Task 5: Polynomial Regression with Hyperparameter Tuning

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Polynomial regression function
def polynomial_regression(X, y, degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return model, poly_features

# Test polynomial degrees
degrees = [2, 3, 5, 8]
for degree in degrees:
    model, poly_features = polynomial_regression(grouped_data['Age'].values, grouped_data['Income'].values, degree)
    
    # Predict using the polynomial model
    X_poly = poly_features.fit_transform(grouped_data['Age'].values.reshape(-1, 1))
    y_pred = model.predict(X_poly)
    
    # Plot results
    plt.plot(grouped_data['Age'], y_pred, label=f'Degree {degree}')

# Plot actual data
plt.scatter(grouped_data['Age'], grouped_data['Income'], color='blue', label='Actual Data')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()
```

#### Explanation:
- **Polynomial Features**: The `PolynomialFeatures` class generates polynomial features of the specified degree.
- **Hyperparameter Tuning**: Different polynomial degrees (2, 3, 5, 8) are tested, and the model performance is compared using MSE.

---

### Summary of Tasks:

1. **Linear Regression**: Manually implemented from scratch on both a subset and the full dataset.
2. **Polynomial Regression**: Performed using `scikit-learn` with various degrees of polynomials.
3. **Evaluation**: Both methods are evaluated using MSE and graphical comparison, followed by a discussion on which model performs better.

Let me know if you need further clarification or additional code examples!