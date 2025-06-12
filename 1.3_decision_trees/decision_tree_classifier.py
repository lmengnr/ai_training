import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess():
    # load data
    fn = os.path.join(os.path.dirname(__file__), "Titanic-Dataset.csv")
    df = pd.read_csv(fn)

    print(df['Age'].shape)

    # basic cleaning
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # drop unused columns
    df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

    # encode categorical features
    df = pd.get_dummies(df, columns=['Sex','Embarked'], drop_first=True)

    # The above call will convert something like this:
    #       Sex Embarked
    # 0    male        S
    # 1  female        C
    # 2  female        Q
    # 3    male        S

    # Into this:
    #    Sex_male  Embarked_Q  Embarked_S
    # 0         1           0           1
    # 1         0           0           0
    # 2         0           1           0
    # 3         1           0           1


    # features/target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y

def main():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=0.05, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
