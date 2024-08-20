### Explanation of Lab 1: Data and Statistics

In this lab, you'll work with data to calculate basic statistics, load a dataset (housing.csv), and visualize insights through plotting. The lab focuses on building a solid foundation in statistical functions and practicing data analysis.

---

### **Task 1: Create Basic Functions**

You need to create several statistical functions from scratch, without using Python's built-in functions like `min()`, `max()`, or `statistics.mean()`. These functions will take a list of grades as input and return statistical values.

Here’s how you can implement these functions:

1. **Min**: Find the minimum value from a list.
2. **Max**: Find the maximum value from a list.
3. **Mean**: Calculate the average of values in a list.
4. **Variance**: Measure the spread of the data from the mean.
5. **Standard Deviation**: Calculate the spread of data relative to the mean.
6. **Median**: Return the middle value of a sorted list.
7. **Median Absolute Deviation**: Calculate the average distance of each value from the median.

### Code for Task 1: Implementing the Statistical Functions

```python
# Min function
def find_min(data):
    min_val = data[0]
    for val in data:
        if val < min_val:
            min_val = val
    return min_val

# Max function
def find_max(data):
    max_val = data[0]
    for val in data:
        if val > max_val:
            max_val = val
    return max_val

# Mean function
def calculate_mean(data):
    total = 0
    for val in data:
        total += val
    return total / len(data)

# Variance function
def calculate_variance(data):
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance

# Standard deviation function
def calculate_std_dev(data):
    variance = calculate_variance(data)
    return variance ** 0.5

# Median function
def calculate_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

# Median absolute deviation function
def calculate_mad(data):
    median = calculate_median(data)
    deviations = [abs(x - median) for x in data]
    return calculate_median(deviations)

# Example usage:
grades = [85, 92, 76, 88, 90, 79, 95, 80]

print("Min:", find_min(grades))
print("Max:", find_max(grades))
print("Mean:", calculate_mean(grades))
print("Variance:", calculate_variance(grades))
print("Standard Deviation:", calculate_std_dev(grades))
print("Median:", calculate_median(grades))
print("Median Absolute Deviation:", calculate_mad(grades))
```

This code defines all the statistical functions that operate on a list of grades. Each function performs the respective calculations.

---

### **Task 2: Get Some Insights from the List of Grades**

1. **Array of Grades**: For a given list of grades, you will calculate the statistics like min, max, mean, standard deviation, median, and median absolute deviation using the functions developed in Task 1.

2. **Explanation of Each Metric**:
   - **Min**: The lowest grade.
   - **Max**: The highest grade.
   - **Mean**: The average score of the students.
   - **Standard Deviation**: A measure of how spread out the grades are from the mean.
   - **Median**: The middle grade when the grades are sorted.
   - **Median Absolute Deviation**: The average distance of grades from the median, showing how much variability there is in the middle of the data.

---

### **Task 3: Load and Inspect the Housing Data**

You will load the `housing.csv` file and inspect its contents. You can use **Pandas** for this task since it simplifies reading and manipulating CSV data.

#### Code for Task 3: Loading and Inspecting Data

```python
import pandas as pd

# Load the dataset
housing_data = pd.read_csv("housing.csv")

# Show the first few rows of the dataset
print(housing_data.head())

# Show information about the dataset
print(housing_data.info())
```

This code loads the housing data and prints out the first few rows to get a sense of the data's structure. The `info()` function provides additional details, such as data types and the number of non-null entries.

---

### **Task 4: Apply Functions on the Real Dataset**

1. **Count the Number of Districts**: You can use the `len()` function to count the number of districts (rows in the dataset).

2. **Calculate the Mean of House Values**: Use your custom `calculate_mean()` function on the column `median_house_value`.

3. **Create Histograms**: For visualizing data, you’ll create histograms using Matplotlib.

#### Code for Task 4: Applying Functions and Creating Histograms

```python
import matplotlib.pyplot as plt

# Task 4.1: Count the number of districts
num_districts = len(housing_data)
print("Number of districts:", num_districts)

# Task 4.2: Calculate the mean of median house values
mean_house_value = calculate_mean(housing_data['median_house_value'].tolist())
print("Mean of house values:", mean_house_value)

# Task 4.3: Create histograms for different columns
housing_data['amount_of_households'].hist(bins=50)
plt.title('Amount of Households')
plt.show()

housing_data['median_income'].hist(bins=50)
plt.title('Median Income')
plt.show()

housing_data['housing_median_age'].hist(bins=50)
plt.title('Housing Median Age')
plt.show()

housing_data['median_house_value'].hist(bins=50)
plt.title('Median House Value')
plt.show()
```

This code creates histograms for the columns `amount_of_households`, `median_income`, `housing_median_age`, and `median_house_value`. By visualizing these distributions, you can identify trends in the data.

---

### **Additional Task: Group by Ocean Proximity**

You are asked to calculate the mean house value for each category in the `ocean_proximity` column.

#### Code for Grouping by Ocean Proximity

```python
# Group by 'ocean_proximity' and calculate mean of 'median_house_value'
mean_house_values_by_proximity = housing_data.groupby('ocean_proximity')['median_house_value'].mean()
print(mean_house_values_by_proximity)
```

This code groups the dataset by the `ocean_proximity` category and calculates the mean house value for each category.

---

### **Analysis of Statistical Metrics**

1. **For Student Grades**:
   - The **mean** is a good indicator of the average performance.
   - The **standard deviation** shows how much the scores vary from the average.
   - The **median** is useful if there are outliers that could skew the mean.

2. **For Income Distribution**:
   - In a dataset with a few extremely high-income individuals (like billionaires), the **median** is often a better metric than the mean because it is less influenced by outliers.

---