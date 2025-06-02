import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Titanic-Dataset.csv")

# Select only the 'Sex' feature and the 'Survived' target, dropping rows with missing values
df = df[['Sex', 'Survived']].dropna()

# Encode 'Sex': male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Define features and target
X = df[['Sex']]
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

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plotting
sex_range = np.array([[0], [1]])
predicted_probs = model.predict_proba(sex_range)[:, 1]

plt.bar(['Male', 'Female'], predicted_probs, color=['blue', 'pink'])
plt.ylabel('Probability of Survival')
plt.title('Logistic Regression: Survival Probability vs. Sex')
plt.savefig('sex_logistic_regression.png')