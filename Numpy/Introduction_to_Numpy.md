# NumPy: The Core Library for Numerical Computing in Python

- NumPy (**Numerical Python**) is a fundamental library for scientific
computing in Python.\
- It provides support for **large multidimensional arrays and matrices**,
along with a collection of **high-level mathematical functions** to
operate on these arrays efficiently.

------------------------------------------------------------------------

## Why NumPy?

-   **Fast**: Much faster than Python lists (written in C).
-   **Efficient**: Uses less memory.
-   **Convenient**: Provides tools for linear algebra, statistics,
    random numbers, etc.
-   **Foundation**: Core library for Pandas, SciPy, Scikit-learn,
    TensorFlow, and many ML/AI frameworks.

------------------------------------------------------------------------

## NumPy Basics

### Importing NumPy

``` python
import numpy as np
```

### Creating Arrays

``` python
arr = np.array([1, 2, 3, 4, 5])
print(arr)
```

### Array Types

-   **1D Array** → `[1, 2, 3]`
-   **2D Array (Matrix)** → `[[1, 2], [3, 4]]`
-   **3D Array (Tensor)** → `[[[1,2],[3,4]], [[5,6],[7,8]]]`

------------------------------------------------------------------------

## Most Important NumPy Methods

### 1. Array Creation

``` python
np.array([1, 2, 3])             # From Python list
np.zeros((2,3))                 # 2x3 matrix of zeros
np.ones((3,3))                  # 3x3 matrix of ones
np.arange(0,10,2)               # [0, 2, 4, 6, 8]
```

### 2. Array Inspection

``` python
arr.shape       # Shape of array (rows, columns)
arr.ndim        # Number of dimensions
arr.size        # Number of elements
arr.dtype       # Data type of array elements
```

### 3. Indexing & Slicing

``` python
arr = np.array([10,20,30,40,50])
print(arr[0])       # First element → 10
print(arr[-1])      # Last element → 50
print(arr[1:4])     # [20, 30, 40]

matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix[0,1])  # Row 0, Col 1 → 2
```

### 4. Mathematical Operations

``` python
arr = np.array([1,2,3,4,5])
print(arr + 10)     # Add scalar → [11,12,13,14,15]
print(arr * 2)      # Multiply scalar
print(np.sqrt(arr)) # Square root of each element in the array
print(np.mean(arr)) # Mean
print(np.std(arr))  # Standard Deviation
```

### 5. Reshaping Arrays

``` python
arr = np.arange(1, 7)       # Creates 1D array: [1 2 3 4 5 6]
print(arr.reshape(2,3))   # Reshape 1D → 2D with 2 rows 3 columns: [[1, 2, 3],[4, 5, 6]]
```

### 6. Combining & Splitting

``` python
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.concatenate((a,b)))      # [1 2 3 4 5 6]

matrix = np.array([[1,2],[3,4],[5,6]])
print(np.split(matrix, 3))        # Split into 3 sub-arrays
```

## Summary of Key Methods

-   **Array Creation:** `array`, `zeros`, `ones`, `arange`, `linspace`
-   **Array Info:** `shape`, `ndim`, `dtype`, `size`
-   **Math Ops:** `+`, `-`, `*`, `/`, `sqrt`, `mean`, `std`, `sum`
-   **Reshape:** `reshape`, `ravel`, `flatten`
-   **Stacking/Splitting:** `concatenate`, `hstack`, `vstack`, `split`
-   **Random:** `rand`, `randn`, `randint`

------------------------------------------------------------------------

## Example-1: Student Scores Analysis with NumPy

``` python
import numpy as np

# Random scores of 10 students (0-100)
scores = np.random.randint(0, 101, 10)

print("Scores:", scores)
print("Average Score:", np.mean(scores))
print("Highest Score:", np.max(scores))
print("Standard Deviation:", np.std(scores))
```

------------------------------------------------------------------------

## Example-2:

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
