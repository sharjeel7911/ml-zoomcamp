# %%
import numpy as np
import pandas as pd

print("Numpy Version: ", np.__version__)
print("Pandas Version: ", pd.__version__)

# %%
# data/car_fuel_efficiency_2026.csv

df = pd.read_csv("data/car_fuel_efficiency_2026.csv")

df.head()

# %%
# How many records are in the dataset?

df.count()

# %%
# How many fuel types are presented in the dataset?

df["fuel_type"].unique()
# %%
# How many columns in the dataset have missing values?

df.isnull().sum()

# %%
# What's the maximum fuel efficiency of cars from Asia?

print(
    "Max efficiency:",
    df[df["origin"].str.lower() == "asia"]["fuel_efficiency_mpg"].max(),
)

# %%
# Find the median value of the horsepower column in the dataset.
# Next, calculate the most frequent value of the same horsepower column.
# Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
# Now, calculate the median value of horsepower once again.

print("Median horsepower:", df["horsepower"].median())
print("Most frequent horsepower:", df["horsepower"].mode()[0])

most_frequent = df["horsepower"].mode()[0]

# %%
# Fill missing horsepower values with the most frequent value
df["horsepower"] = df["horsepower"].fillna(most_frequent)

# %%
print(df["horsepower"].median())
# %%

Select all the cars from Asia
Select only columns vehicle_weight and model_year
Select the first 7 values
Get the underlying NumPy array. Let's call it X.

Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
Invert XTX.
Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
What's the sum of all the elements of the result?

# %%

X = df[df["origin"].str.lower() == "asia"][["vehicle_weight", "model_year"]].head(7)

X = X.values
print(X)
X.shape
type(X)

# %%

def matrix_transpose(U: np.ndarray) -> np.ndarray
	rows = U.shape[0]
	cols = U.shape[1]
	result = np.zeros((cols, rows))

	for i in range(rows):
		for j in range(cols):
			result[j][i] = U[i][j]
	return result
