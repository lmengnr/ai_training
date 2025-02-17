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

Y = df_cleaned['MEDV']


params = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

results = dict()

for parameter in params:
    
    X = df_cleaned[[parameter]]


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X, Y)

    Y_pred = model.predict(X_test)

    mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
    r2 = r2_score(y_true=Y_test, y_pred=Y_pred)

    F_score, p_value = f_regression(X, Y)

    metrics = dict()

    metrics["r2"] = r2
    metrics["f_score"] = F_score[0]
    metrics["p_value"] = p_value[0]

    results[parameter] = metrics


print(results)

df_results = pd.DataFrame.from_dict(results, orient="index")

# Create figure and hide axes
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("tight")
ax.axis("off")

# Create table
table = ax.table(cellText=df_results.round(4).values,
                 colLabels=df_results.columns,
                 rowLabels=df_results.index,
                 cellLoc="center",
                 loc="center")

plt.savefig("multiple_regression_results.png")