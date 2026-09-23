# %%
import numpy as np
import pandas as pd

print(np.__version__)
print(pd.__version__)

# ====================================================================================================
# ====================================================================================================

# %%
# Numpy
# Numpy arrays
np.zeros(10)
np.full(10, 2.5)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(arr)

np.arange(5, 10)  # it generates an array of numbers from 5 to 9
print(np.arange(5, 10))

two_d = np.zeros((3, 3))
print(two_d.shape)
print(two_d.size)

two_d[1, 2] = 99
print(two_d)
print()

two_d[:, 0] = [67, 77, 87]  # set the first column to 77
print(two_d)
print()

two_d[2] = 88  # set the third row to 88
print(two_d)

# %%
np.random.rand(3, 3)
print(np.random.randint(low=0, high=100, size=(3, 3)))

# %%
# Element-wise operations
a = np.arange(5)
print(a, "\n")
b = (10 + (a * 2)) ** 2 / 100
print(a / b + 10, "\n")

print(a >= 2, "\n")
print(a > b, "\n")
print(a[a > b])  # give the elements of a that are greater than b

# ====================================================================================================
# ====================================================================================================

# %%
# Linear algebra

# Vector-Vector multiplication

u = np.array([2, 4, 5, 6])
v = np.array([1, 0, 0, 2])

print(v.shape[0])


def vector_multiplication(u: np.ndarray, v: np.ndarray) -> float:
    assert u.shape[0] == v.shape[0]

    result = 0.0
    n = u.shape[0]

    for i in range(n):
        result = result + u[i] * v[i]

    return result


print("Manual: ", vector_multiplication(u, v))
print("Builtin: ", u.dot(v))


# ====================================================================================================
# ====================================================================================================

# %%
# Matrix-Vector multiplication

v = np.array([1, 1, 1, 2])
U = np.array(
    [
        [2, 4, 5, 6],
        [1, 2, 1, 2],
        [3, 1, 2, 1],
    ]
)

print(U.shape, v.shape)


def matrix_vector_multiplication(U: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert U.shape[1] == v.shape[0]

    m = U.shape[0]  # 3 rows
    n = U.shape[1]  # 4 cols

    result = np.zeros(m)

    for i in range(m):
        for j in range(n):
            result[i] = result[i] + U[i][j] * v[j]

    return result


print("Manual: ", matrix_vector_multiplication(U, v))
print("Builtin: ", U.dot(v))

# def matrix_vector_multiplication(U, v):
#     assert U.shape[1] == v.shape[0]

#     num_rows = U.shape[0]

#     result = np.zeros(num_rows)

#     for i in range(num_rows):
#         result[i] = vector_vector_multiplication(U[i], v)

#     return result

# ====================================================================================================
# ====================================================================================================

# %%
# Matrix-Matrix multiplication
U = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
V = np.array(
    [
        [1, 1, 2],
        [0, 0.5, 1],
        [0, 2, 1],
        [2, 1, 0],
    ]
)

print(U.shape, V.shape)


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


print("Manual: ", matrix_multiplication(U, V))
print("Builtin: ", U.dot(V))

# def matrix_matrix_multiplication(U, V):
#     assert U.shape[1] == V.shape[0]

#     num_rows = U.shape[0]
#     num_cols = V.shape[1]

#     result = np.zeros((num_rows, num_cols))

#     for i in range(num_cols):
#         vi = V[:, i]
#         Uvi = matrix_vector_multiplication(U, vi)
#         result[:, i] = Uvi

#     return result

# ====================================================================================================
# ====================================================================================================

# %%
# Identity Matrix

I = np.eye(5)
print(I, "\n")

# Inverse Matrix

# (A^−1)A = A(A^−1) = I

A = np.array([[1, 2], [3, 4]])

A_inv = np.linalg.inv(A)
print(A_inv)

# ====================================================================================================
# ====================================================================================================

# %%
# Pandas

# data
data = [
    ["Nissan", "Stanza", 1991, 138, 4, "MANUAL", "sedan", 2000],
    ["Hyundai", "Sonata", 2017, None, 4, "AUTOMATIC", "Sedan", 27150],
    ["Lotus", "Elise", 2010, 218, 4, "MANUAL", "convertible", 54990],
    ["GMC", "Acadia", 2017, 194, 4, "AUTOMATIC", "4dr SUV", 34450],
    ["Nissan", "Frontier", 2017, 261, 6, "MANUAL", "Pickup", 32340],
]

columns = [
    "Make",
    "Model",
    "Year",
    "Engine HP",
    "Engine Cylinders",
    "Transmission Type",
    "Vehicle_Style",
    "MSRP",
]

# dicts
# data = [
#     {
#         "Make": "Nissan",
#         "Model": "Stanza",
#         "Year": 1991,
#         "Engine HP": 138.0,
#         "Engine Cylinders": 4,
#         "Transmission Type": "MANUAL",
#         "Vehicle_Style": "sedan",
#         "MSRP": 2000
#     },
#     {
#         "Make": "Hyundai",
#         "Model": "Sonata",
#         "Year": 2017,
#         "Engine HP": None,
#         "Engine Cylinders": 4,
#         "Transmission Type": "AUTOMATIC",
#         "Vehicle_Style": "Sedan",
#         "MSRP": 27150
#     },
#     {
#         "Make": "Lotus",
#         "Model": "Elise",
#         "Year": 2010,
#         "Engine HP": 218.0,
#         "Engine Cylinders": 4,
#         "Transmission Type": "MANUAL",
#         "Vehicle_Style": "convertible",
#         "MSRP": 54990
#     },
#     {
#         "Make": "GMC",
#         "Model": "Acadia",
#         "Year": 2017,
#         "Engine HP": 194.0,
#         "Engine Cylinders": 4,
#         "Transmission Type": "AUTOMATIC",
#         "Vehicle_Style": "4dr SUV",
#         "MSRP": 34450
#     },
#     {
#         "Make": "Nissan",
#         "Model": "Frontier",
#         "Year": 2017,
#         "Engine HP": 261.0,
#         "Engine Cylinders": 6,
#         "Transmission Type": "MANUAL",
#         "Vehicle_Style": "Pickup",
#         "MSRP": 32340
#     }
# ]

# data load
df = pd.DataFrame(data, columns=columns)
df

df.head(n=3)

# %%
# Shows columns
df.Make
df["Engine HP"]
df[["Make", "Model", "Year"]]

# %%
# Add & remove columns
df["Id"] = [1, 2, 3, 4, 5]

del df["Id"]

print(df.index)
# %%
# Lookup by index -> Returns 2 rows
# df.loc[[3, 4]]
df.index = ["a", "b", "c", "d", "e"]
df.loc[["d", "e"]]
# df.loc[[3, 4]] ->  it wont work because index is not 3, 4 but 'c', 'd'


# %%
# Diff between loc and iloc -> loc uses labels, iloc uses integer positions(0.....)
df.iloc[[3, 4]]

# %%
df = (
    df.reset_index()
)  # resets the index to default integer positions but keeps the old index as a column
df = df.reset_index(
    drop=True
)  # resets the index to default integer positions and drops the old index

# %%

# Element wise operations
df["Engine HP"] + 4
df.groupby("Transmission Type").MSRP.min()
df.groupby("Transmission Type").MSRP.max()

# %%
df[
    df["Make"] == "Nissan"
]  # returns a new DataFrame with only the rows where 'Make' is 'Nissan'
"machine learning zoomcamp".replace(" ", "_")
df["Vehicle_Style"].str.lower()  # returns a new Series with lowercased values
df["Vehicle_Style"] = (
    df["Vehicle_Style"].str.replace(" ", "_").str.lower()
)  # assigns the lowercased values back to the 'Vehicle_Style' column
df.describe().round(2)  # returns a new DataFrame with rounded values
df.nunique()  # returns a Series with the number of unique values for each column
df.isnull().sum()  # returns a Series with the number of null values for each column
df.MSRP.values  # returns the values of the 'MSRP' column as a NumPy array
df.to_dict(
    orient="records"
)  # returns a list of dictionaries, where each dictionary represents a row
