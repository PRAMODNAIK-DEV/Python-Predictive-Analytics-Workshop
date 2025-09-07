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

### 2. DataFrame

Two-dimensional labeled data structure (rows + columns).

``` python
data = {"Name": ["Pramod", "Ravi", "Sneha"],
        "Age": [25, 30, 28],
        "Salary": [50000, 60000, 55000]}
df = pd.DataFrame(data)
print(df)
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
df["column"]                # Select a column
df[["col1", "col2"]]        # Select multiple columns
df.iloc[rows, cols]         # Select by index
df.loc[rows, cols]          # Select by labels
```

### 3. Filtering & Conditional Selection

``` python
df[df["Age"] > 25]
df[(df["Age"] > 25) & (df["Salary"] > 55000)]
```

### 4. Handling Missing Data

-   `df.isnull()` → Check missing values
-   `df.dropna()` → Remove missing rows
-   `df.fillna(value)` → Fill missing values

### 5. Adding & Modifying Columns

``` python
df["Bonus"] = df["Salary"] * 0.1
df.rename(columns={"Name": "Employee"}, inplace=True)
```

### 6. Sorting

``` python
df.sort_values("Salary", ascending=False)
df.sort_index()
```

### 7. Aggregation & Grouping

``` python
df.groupby("Age")["Salary"].mean()
df["Salary"].agg(["min", "max", "mean"])
```

### 8. Merging, Joining, Concatenation

``` python
pd.concat([df1, df2], axis=0)
pd.merge(df1, df2, on="ID", how="inner")
```

### 9. Exporting Data

-   `df.to_csv("data.csv", index=False)`
-   `df.to_excel("data.xlsx", index=False)`
-   `pd.read_csv("data.csv")`
-   `pd.read_excel("data.xlsx")`

------------------------------------------------------------------------

## To be continued: [Predictive_Analysis_using_Pandas](Predictive_Analysis_using_Pandas.md)