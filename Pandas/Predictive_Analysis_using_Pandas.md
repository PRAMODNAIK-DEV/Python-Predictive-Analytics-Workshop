# Predictive Analysis using Pandas & Logistic Regression

This example demonstrates how to use **Pandas** and **Scikit-learn** to
build a simple predictive model: predicting whether a student passes an
exam based on study hours.

------------------------------------------------------------------------

## Step 1: Import Libraries

``` python
import pandas as pd
from sklearn.linear_model import LogisticRegression
```

-   **pandas** → for handling tabular data (creating DataFrame).
-   **LogisticRegression** → from scikit-learn, used for binary
    classification (Pass/Fail).

------------------------------------------------------------------------

## Step 2: Create Sample Data

``` python
data = pd.DataFrame({
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'passed':      [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
})
```

-   We create a **DataFrame** with two columns:
    -   `study_hours`: Number of hours studied.
    -   `passed`: Target variable → `0` (Fail), `1` (Pass).

👉 The assumption is: the more you study, the higher chance of passing.

------------------------------------------------------------------------

## Step 3: Define Features and Target

``` python
X = data[['study_hours']]  # Feature (independent variable)
y = data['passed']         # Target (dependent variable)
```

-   **X** → Input feature(s) → `study_hours`.
-   **y** → Output/label → `passed`.

------------------------------------------------------------------------

## Step 4: Train the Model

``` python
model = LogisticRegression()
model.fit(X, y)
```

-   Initialize a **Logistic Regression** model.
-   Train (`fit`) the model using study hours (X) to predict pass/fail
    (y).

------------------------------------------------------------------------

## Step 5: Make Predictions

``` python
print("Prediction (7 hours):", model.predict([[7]]))  # Expected: Pass
print("Prediction (2 hours):", model.predict([[2]]))  # Expected: Fail
```

-   Predict whether a student passes given study hours:
    -   **7 hours → likely Pass (`1`)**
    -   **2 hours → likely Fail (`0`)**

------------------------------------------------------------------------

## Summary

-   **Pandas** is used for creating and managing the dataset.
-   **Scikit-learn (LogisticRegression)** is used to train a predictive
    model.
-   The model learns the relationship between study hours and exam
    results.
-   We can then **predict outcomes for new students** based on study
    hours.
