import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Titanic-Dataset.csv")

print(df.shape)


# Select only the 'Fare' feature and the 'Survived' target, dropping rows with missing values
df = df[['Fare', 'Survived']].dropna()

# Define features and target
X = df[['Fare']]  # features must be 2D
y = df['Survived']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt

# Create a range of 'Fare' values
fare_range = np.linspace(X['Fare'].min(), X['Fare'].max(), 300).reshape(-1, 1)

# Predict probabilities for the range
predicted_probs = model.predict_proba(fare_range)[:, 1]

# Plot the original data points
plt.scatter(X, y, color='gray', alpha=0.5, label='Data points')

# Plot the logistic regression curve
plt.plot(fare_range, predicted_probs, color='red', label='Logistic Regression Curve')

plt.xlabel('Fare')
plt.ylabel('Probability of Survival')
plt.title('Logistic Regression: Survival Probability vs. Fare')
plt.legend()
plt.savefig('single_logistic_regression.png')

