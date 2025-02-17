import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression

df = pd.read_csv("HousingData.csv")

print(df.head())

PARAM = "RM"
PARAM2 = "LSTAT"

# Remove NA values
df_cleaned = df.dropna()

X = df_cleaned[[PARAM]]
Y = df_cleaned['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X, Y)

Y_pred = model.predict(X_test)

mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
r2 = r2_score(y_true=Y_test, y_pred=Y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient ({PARAM}): {model.coef_[0]:.2f}")

F_score, p_value = f_regression(X, Y)

print(f"F-score: {F_score[0]:.4f}")
print(f"P-value: {p_value[0]:.4f}")


plt.scatter(X_test, Y_test, color='blue', label="Actual Data")
plt.plot(X_test, Y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel(f"{PARAM}")
plt.ylabel("MEDV (Median House Price in $1000s)")
plt.title(f"Linear Regression: {PARAM} vs. MEDV")
plt.legend()
# plt.show()

plt.savefig("regression_plot.png")  # Saves as an image file
print("Plot saved as regression_plot.png")


# -------------------------------------
# Model 2: Using two features
# -------------------------------------
PARAMS2 = [PARAM, PARAM2]
X2 = df_cleaned[PARAMS2]

# Split data for model 2
X2_train, X2_test, _, _ = train_test_split(X2, Y, test_size=0.2, random_state=42)

# Train model 2
model2 = LinearRegression()
model2.fit(X2_train, Y_train)
Y2_pred = model2.predict(X2_test)

# Evaluate model 2
mse2 = mean_squared_error(Y_test, Y2_pred)
r2_2 = r2_score(Y_test, Y2_pred)

print(f"\nModel 2 (Two Features: {PARAM} and {PARAM2})")
print(f"Mean Squared Error (MSE): {mse2:.2f}")
print(f"R-squared (R²): {r2_2:.2f}")
print(f"Intercept: {model2.intercept_:.2f}")
print(f"Coefficients: {dict(zip(PARAMS2, model2.coef_))}")

F_score2, p_value2 = f_regression(X2, Y)
for feature, f_val, p_val in zip(PARAMS2, F_score2, p_value2):
    print(f"Feature: {feature}, F-score: {f_val:.4f}, p-value: {p_val:.4f}")

# Plot for Model 2: MEDV vs PARAM using predicted values from model2
plt.figure(figsize=(10, 6))

# Scatter plot of actual values
plt.scatter(X2_test[PARAM], Y_test, color='blue', label="Actual MEDV", alpha=0.7)

# For a line plot, sort by PARAM so the line looks smooth
sorted_idx = X2_test[PARAM].argsort()
rm_sorted = X2_test[PARAM].iloc[sorted_idx]
pred_sorted = Y2_pred[sorted_idx]

# Plot the regression line (predicted values from model2)
plt.plot(rm_sorted, pred_sorted, color='red', linewidth=2, label="Regression Line (Model 2)")

plt.xlabel(f"{PARAM}")
plt.ylabel("MEDV (Median House Price in $1000s)")
plt.title(f"Model 2: Linear Regression ({PARAM} vs. MEDV) using Two Features")
plt.legend()
plt.savefig(f"regression_plot_model2_{PARAM}.png")
print(f"Model 2 {PARAM} plot saved as regression_plot_model2_{PARAM}.png")


import scipy.stats as stats

# Number of predictors in each model
p_reduced = 1   # model1 has one predictor
p_full = 2      # model2 has two predictors

# Number of test samples
n = len(Y_test)

# Calculate Sum of Squared Errors for each model
SSE_reduced = np.sum((Y_test - Y_pred) ** 2)
SSE_full = np.sum((Y_test - Y2_pred) ** 2)

# Compute the F-statistic
numerator = (SSE_reduced - SSE_full) / (p_full - p_reduced)
denominator = SSE_full / (n - p_full - 1)
F_stat = numerator / denominator

# Calculate the p-value using the F-distribution survival function
p_val = stats.f.sf(F_stat, p_full - p_reduced, n - p_full - 1)

print(f"F-statistic for comparing Model1 and Model2: {F_stat:.4f}")
print(f"P-value for the extra feature: {p_val:.4f}")
