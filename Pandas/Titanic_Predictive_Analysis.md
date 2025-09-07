# Predictive Analysis on Titanic Dataset (Logistic Regression)

This example demonstrates how to use **Pandas**, **Seaborn**, and
**Scikit-learn** to perform predictive analysis on the Titanic dataset.\
We predict whether a passenger survived based on features such as age,
fare, sex, and class.

------------------------------------------------------------------------

## Step 1: Install and Import Libraries

```bash
pip install seaborn scikit-learn matplotlib pandas numpy
```

``` python
import seaborn as sns           # Seaborn comes with some built-in sample datasets (like Titanic, Iris, Tips, Penguins, etc.)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

-   **seaborn** → to load Titanic dataset easily.
-   **train_test_split** → to split data into training & testing sets.
-   **LogisticRegression** → for binary classification (survived or
    not).
-   **accuracy_score** → to evaluate the model.

------------------------------------------------------------------------

## Step 2: Load Data

``` python
df = sns.load_dataset("titanic").dropna(subset=['age','fare','class','sex','survived'])
```

-   Loads Titanic dataset from Seaborn.
-   Drops rows with missing values in critical columns.

------------------------------------------------------------------------

## Step 3: Preprocessing

``` python
df['sex'] = df['sex'].map({'male':0, 'female':1})
X = df[['age','fare','sex','pclass']]
y = df['survived']
```

-   Convert `sex` column to numeric (`male=0`, `female=1`).
-   Define **features (X)**: age, fare, sex, passenger class (pclass).
-   Define **target (y)**: survived (0 = No, 1 = Yes).

------------------------------------------------------------------------

## Step 4: Train-Test Split

``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- `test_size=0.3` → Split dataset into **80% training** and **20% testing**.
-   Ensures model is trained on one portion and evaluated on unseen
    data. 
- `random_state=42` → ensures that every time we run the code, we get the same train and test split there won't be a shuffling. This is used to control the randomness of how the data is split. The number 42 is just a number — you could use 0, 1, 99, or any other integer. 

------------------------------------------------------------------------

## Step 5: Train Logistic Regression Model

``` python
model = LogisticRegression()
model.fit(X_train, y_train)
```

-   Initialize Logistic Regression model.
-   Train using training dataset.

------------------------------------------------------------------------

## Step 6: Make Predictions

``` python
y_pred = model.predict(X_test)
```

-   Predict survival for the test set.

------------------------------------------------------------------------

## Step 7: Evaluate Model

``` python
print("Accuracy:", accuracy_score(y_test, y_pred))
```

-   Compare predicted vs actual values.
-   **Accuracy score** shows how well the model performed.

------------------------------------------------------------------------

## Summary

-   Logistic Regression is applied on Titanic dataset.
-   Features considered: **Age, Fare, Sex, Passenger Class**.
-   Target variable: **Survived (0 or 1)**.
-   After preprocessing & training, we can measure prediction accuracy
    on test data.

---
## To be continued: [Introduction_to_Numpy](../Numpy/Introduction_to_Numpy.md)
