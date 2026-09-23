# %%
import numpy as np
import pandas as pd

# ====================================================================================================

# %%
# Read "data/car_fuel_efficiency_2026.csv"
df = pd.read_csv("data/car_fuel_efficiency_2026.csv")
df.head()

# ====================================================================================================

# %%
# Q1. Pandas version
# What version of Pandas did you install?
print("Numpy Version: ", np.__version__)
print("Pandas Version: ", pd.__version__)

# ====================================================================================================

# %%
# Q2. Records count
# How many records are in the dataset?
print("Number of records:", df.shape[0])  # shape gives (rows, columns)

# ====================================================================================================

# %%
# Q3. Fuel types
# How many fuel types are presented in the dataset?
print(df["fuel_type"].unique())
print()
print("Number of fuel types:", df["fuel_type"].nunique())

# ====================================================================================================

# %%
# Q4. Missing values
# How many columns in the dataset have missing values?
print(df.isnull().sum())
print()
print("Columns with missing values:", (df.isnull().sum() > 0).sum())

# ====================================================================================================

# %%
# Q5. Max fuel efficiency
# What's the maximum fuel efficiency of cars from Asia?
print(
    "Max efficiency:",
    df[df["origin"].str.lower() == "asia"]["fuel_efficiency_mpg"].max(),
)

# ====================================================================================================

# %%
# Q6. Median value of horsepower
# 1. Find the median value of the horsepower column in the dataset.
# 2. Next, calculate the most frequent value of the same horsepower column.
# 3. Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
# 4. Now, calculate the median value of horsepower once again.

# Median & Mode
print("Median horsepower:", df["horsepower"].median())
print("Most frequent horsepower:", df["horsepower"].mode()[0])

# %%
# Fill missing horsepower values with the most frequent value
most_frequent = df["horsepower"].mode()[0]
df["horsepower"] = df["horsepower"].fillna(most_frequent)

# %%
# Median of horsepower after filling missing values
print("Median horsepower after filling missing values:", df["horsepower"].median())

# ====================================================================================================

# %%
# Q7. Sum of weights
# 1. Select all the cars from Asia
# 2. Select only columns vehicle_weight and model_year
# 3. Select the first 7 values
# 4. Get the underlying NumPy array. Let's call it X.
# 5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# 6. Invert XTX.
# 7. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
# 8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# 9. What's the sum of all the elements of the result?

# 1, 2, 3
X = df[df["origin"].str.lower() == "asia"][["vehicle_weight", "model_year"]].head(7)

# 4. Convert to NumPy array
X = X.values
print(X)
print("X.shape:", X.shape)


# %%
# 5. Transpose
def matrix_transpose(U: np.ndarray) -> np.ndarray:
    rows = U.shape[0]
    cols = U.shape[1]
    result = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            result[j][i] = U[i][j]
    return result


print("Builtin:", X.T)
print()
print("Manual:", matrix_transpose(X))
print()

T = matrix_transpose(X)
print("T:", T)


# %%
# Matrix Multiplication
def matrix_multiplication(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    assert U.shape[1] == V.shape[0]

    rows = U.shape[0]  # 3
    cols = V.shape[1]  # 3
    common = U.shape[1]  # 4

    result = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            for k in range(common):
                result[i][j] += U[i][k] * V[k][j]

    return result


print("Builtin:", T.dot(X))
print()
print("Manual:", matrix_multiplication(T, X))
print()

XTX = matrix_multiplication(T, X)
print("XTX:", XTX)

# %%
# 6. Invert XTX
XTX_inv = np.linalg.inv(XTX)
print("XTX inverse:", XTX_inv)


# %%
# 7, 8
def matrix_vector_multiplication(U: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert U.shape[1] == v.shape[0]

    m = U.shape[0]  # 3 rows
    n = U.shape[1]  # 4 cols

    result = np.zeros(m)

    for i in range(m):
        for j in range(n):
            result[i] = result[i] + U[i][j] * v[j]

    return result


y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = matrix_multiplication(XTX_inv, T)
w = matrix_vector_multiplication(w, y)
print("w:", w)

# %%
# 9. Sum of all elements
answer = w.sum()
print("Sum:", answer)
