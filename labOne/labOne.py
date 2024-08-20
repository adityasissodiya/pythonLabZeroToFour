# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

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
    variance = sum((x - mean) ** 2 for x in data) / len(data
    )
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

# Example usage of statistical functions with sample data
grades = [85, 92, 76, 88, 90, 79, 95, 80]

print("Min:", find_min(grades))
print("Max:", find_max(grades))
print("Mean:", calculate_mean(grades))
print("Variance:", calculate_variance(grades))
print("Standard Deviation:", calculate_std_dev(grades))
print("Median:", calculate_median(grades))
print("Median Absolute Deviation:", calculate_mad(grades))

# Load the dataset
housing_data = pd.read_csv("/home/aditya/Documents/pythonLabZeroToFour/labOne/housing.csv")

# Show the first few rows of the dataset
print(housing_data.head())

# Show information about the dataset
print(housing_data.info())

# Task 4.1: Count the number of districts (or rows in the dataset)
num_districts = len(housing_data)
print("Number of districts:", num_districts)

# Task 4.2: Calculate the mean of median house values
mean_house_value = calculate_mean(housing_data['median_house_value'].tolist())
print("Mean of house values:", mean_house_value)

# Task 4.3: Create histograms for different columns
housing_data['total_rooms'].hist(bins=50)
plt.title('Total Rooms')
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

# Task 4.4: Group by 'ocean_proximity' and calculate mean of 'median_house_value'
mean_house_values_by_proximity = housing_data.groupby('ocean_proximity')['median_house_value'].mean()
print(mean_house_values_by_proximity)
