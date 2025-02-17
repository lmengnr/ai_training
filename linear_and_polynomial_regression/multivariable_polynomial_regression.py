"""
In this script a polynomial regression model using 4 variables is trained.
A scatter plot is generated that plots the test values and the predicted values of the model

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression


df = pd.read_csv("HousingData.csv")


# Remove NA values
df_cleaned = df.dropna()

selected_features = ['LSTAT', 'RM', 'INDUS', 'PTRATIO']
X = df_cleaned[selected_features]
Y = df_cleaned['MEDV']

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)

# Train Polynomial Regression Model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Evaluate
r2_poly = r2_score(Y_test, Y_pred)
print(f"Polynomial Regression R² Score: {r2_poly:.4f}")

f_values, p_values = f_regression(X_train, Y_train)

for feature, f_val, p_val in zip(poly.get_feature_names_out(selected_features), f_values, p_values):
    print(f"Feature: {feature}, F-value: {f_val:.4f}, p-value: {p_val:.4f}")

# Create a new figure
plt.figure(figsize=(10, 6))

# Plot actual MEDV values in red
plt.scatter(range(len(Y_test)), Y_test, color='red', label='Actual MEDV', alpha=0.7)

# Plot predicted MEDV values in blue
plt.scatter(range(len(Y_test)), Y_pred, color='blue', label='Predicted MEDV', alpha=0.7)

# Add labels and title
plt.xlabel("Sample Index")
plt.ylabel("MEDV")
plt.title("Actual vs. Predicted MEDV Values")
plt.legend()


plt.savefig("polynomial_regression.png")

