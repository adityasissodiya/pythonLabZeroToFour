# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def load_csv(filepath):
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skipping the header
            values = line.strip().split(',')
            # Handle mixed data types: convert only numeric fields to float
            row = []
            for v in values:
                try:
                    # Try to convert to float
                    row.append(float(v))
                except ValueError:
                    # If conversion fails, keep as a string
                    row.append(v)
            data.append(row)
    return data

# Loading the subset data from the CSV
subset_data = load_csv('C:/Users/adisis/OneDrive - Luleå University of Technology/Documents/pythonLabZeroToFour/labTwo/inc_subset.csv')

# Inspect data using print or debugger
for row in subset_data:
    print(row)  # Print out data to verify it has loaded correctly

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

# Load the full dataset
full_data = load_csv('C:/Users/adisis/OneDrive - Luleå University of Technology/Documents/pythonLabZeroToFour/labTwo/inc_utf.csv')

# Convert to DataFrame
df_full = pd.DataFrame(full_data, columns=['Index', 'Region', 'Age', 'Income'])


# Clean the Age column by replacing 'years' and handling '100+' and other non-numeric cases
def clean_age(age_str):
    # Remove the ' years' part
    age_str = age_str.replace(' years', '')

    # Handle '100+' by replacing it with 100
    if age_str == '100+':
        return 100

    # Try to convert to integer, return NaN if conversion fails
    try:
        return int(age_str)
    except ValueError:
        return np.nan


# Apply the cleaning function to the Age column
df_full['Age'] = df_full['Age'].apply(clean_age)

# Convert the Income column to numeric, forcing non-numeric values to NaN
df_full['Income'] = pd.to_numeric(df_full['Income'], errors='coerce')

# Drop any rows where Age or Income is NaN
df_full = df_full.dropna(subset=['Age', 'Income'])

# Ensure that only numeric columns are used in the groupby and mean calculation
grouped_data = df_full.groupby('Age')['Income'].mean().reset_index()


# Perform linear regression on the grouped data
def linear_regression(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    # Using the formula for slope (m) and intercept (b)
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * X_mean

    return slope, intercept


# Perform linear regression
slope_full, intercept_full = linear_regression(grouped_data['Age'], grouped_data['Income'])

# Plot the results
regression_line_full = slope_full * grouped_data['Age'] + intercept_full

plt.scatter(grouped_data['Age'], grouped_data['Income'], color='blue', label='Grouped Data')
plt.plot(grouped_data['Age'], regression_line_full, color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()

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
