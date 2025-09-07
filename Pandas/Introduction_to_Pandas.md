# Pandas in Python - Detailed Guide

Pandas is one of the most important Python libraries for **data analysis
& predictive analytics**.

------------------------------------------------------------------------

## Pandas Data Structures

### 1. Series

One-dimensional labeled array.

``` python
import pandas as pd

s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])
print(s)
```
Output:
```bash
a    10
b    20
c    30
d    40
dtype: int64
```
Here,
- `dtype` → stands for data type of the elements in your Pandas Series.
- `int64` → means each element in your series is stored as a 64-bit integer.

---
### 2. DataFrame

Two-dimensional labeled data structure (rows + columns).

``` python
data = {"Name": ["Pramod", "Ravi", "Sneha"],
        "Age": [25, 30, 28],
        "Salary": [50000, 60000, 55000]}
df = pd.DataFrame(data)
print(df)
```
Output:
```bash
     Name  Age  Salary
0  Pramod   25   50000
1    Ravi   30   60000
2   Sneha   28   55000
```
------------------------------------------------------------------------

## Most Important Pandas Functions & Methods

### 1. Data Exploration

-   `df.head(n)` → First n rows
-   `df.tail(n)` → Last n rows
-   `df.shape` → Dimensions (rows, columns)
-   `df.info()` → Column data types & null values
-   `df.describe()` → Summary statistics
-   `df.columns` → Column names
-   `df.index` → Row index labels

### 2. Data Selection

``` python
df["Name"]                # Select a column
df[["Name", "Age"]]        # Select multiple columns
df.iloc[rows, cols]         # Select by index rows and columns df.iloc[0,1] 0th Row and 1st column, df.iloc[0:2, 1:3] 0 to 1 row and 1 to 2 column.
df.loc[rows, cols]          # Select by labels. df.loc[0, "Name"] --> Row label 0, column "Name" → Alice
```

### 3. Filtering & Conditional Selection

``` python
df[df["Age"] > 25]
df[(df["Age"] > 25) & (df["Salary"] > 55000)]
```

### 4. Handling Missing Data

-   `df.isnull()` → Check missing values
-   `df.dropna()` → Remove Rows with Missing Values
-   `df.fillna(value)` → Fill missing values. df.fillna(value={"Age": 0, "City": "Unknown"})

### 5. Adding & Modifying Columns

``` python
df["Bonus"] = df["Salary"] * 0.1
df.rename(columns={"Name": "Employee"}, inplace=True)
```
Here
- `inplace=True` : is to tell pandas to apply the change directly to the original DataFrame, without creating a new copy.
### 6. Sorting

``` python
df.sort_values("Salary", ascending=False)
df.sort_index()
```

### 7. Aggregation & Grouping

``` python
df.groupby("Age")["Salary"].mean()       # It groups the DataFrame by Age and then calculates the average Salary for each age group.
df["Salary"].agg(["min", "max", "mean"]) # This tells Pandas to calculate the minimum, maximum, and average of the Salary column.
```

### 8. Merging, Joining, Concatenation

``` python
pd.concat([df1, df2], axis=0)           # Concatenates (stacks) two DataFrames on top of each other (row-wise, since axis=0).
pd.merge(df1, df2, on="ID", how="inner")        # merge is just like SQL JOIN.
```

### 9. Exporting Data

-   `df.to_csv("data.csv", index=False)`
-   `df.to_excel("data.xlsx", index=False)`
-   `pd.read_csv("data.csv")`
-   `pd.read_excel("data.xlsx")`

------------------------------------------------------------------------

## To be continued: [Predictive_Analysis_using_Pandas](Predictive_Analysis_using_Pandas.md)