import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression

df = pd.read_csv("HousingData.csv")

print(df.head())

# Remove NA values
df_cleaned = df.dropna()

X = df_cleaned[['CRIM']]
Y = df_cleaned['MEDV']

# print(X)
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X, Y)

Y_pred = model.predict(X_test)

mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
r2 = r2_score(y_true=Y_test, y_pred=Y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient (CRIM): {model.coef_[0]:.2f}")

F_score, p_value = f_regression(X, Y)

print(f"F-score: {F_score[0]:.4f}")
print(f"P-value: {p_value[0]:.4f}")


plt.scatter(X_test, Y_test, color='blue', label="Actual Data")
plt.plot(X_test, Y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel("CRIM (Per Capita Crime Rate)")
plt.ylabel("MEDV (Median House Price in $1000s)")
plt.title("Linear Regression: CRIM vs. MEDV")
plt.legend()
# plt.show()

plt.savefig("regression_plot.png")  # Saves as an image file
print("Plot saved as regression_plot.png")
