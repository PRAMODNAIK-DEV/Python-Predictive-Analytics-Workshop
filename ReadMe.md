# Workshop: Python Libraries for Predictive Analysis (NumPy & Pandas)

## Agenda

### **Part 1: Introduction & Setup**

-   What is Predictive Analysis?
-   Role of NumPy & Pandas in Predictive Analytics
-   Environment Setup


------------------------------------------------------------------------
### **Part 2: Pandas for Predictive Analysis**
- Introduction to Pandas? **[Introduction_to_Pandas.md](Introduction_to_Pandas.md)**
- Predictive Analysis using Pandas
- Simple Predictive Model Using Pandas & ML
- Hands-On
    - **[Predictive_Analysis_using_Pandas](Predictive_Analysis_using_Pandas.md)**
    - **[Titanic_Predictive_Analysis](Titanic_Predictive_Analysis.md)**


### **Part 3: NumPy for Predictive Analysis**

#### Example

``` python
import numpy as np

# Array basics
arr = np.array([10, 20, 30, 40])
print("Mean:", arr.mean())
print("Standard Deviation:", arr.std())

# Slicing & Broadcasting
arr = np.arange(1, 11)
print(arr[2:7])
print(arr * 2)
print(arr[arr > 5])

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Dot Product:\n", np.dot(A, B))

# Random data generation
data = np.random.normal(50, 10, 1000)
print("Sample Mean:", np.mean(data))
```

#### Activities

-   Generate 100 random exam scores (0--100)
-   Find mean, median, max, min
-   Count how many scored \> 75

------------------------------------------------------------------------

