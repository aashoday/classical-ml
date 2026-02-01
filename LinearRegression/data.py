# Import Libraries 
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Fetch data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Split data in training and testing sets
SPLIT_POINT = int(0.8 * len(X))
X_train, y_train = X.iloc[:SPLIT_POINT], y.iloc[:SPLIT_POINT] 
X_test, y_test = X.iloc[SPLIT_POINT:], y.iloc[SPLIT_POINT:]

# Print length and shape of splits 
print(f"Length of X_train: {len(X_train)} | Shape of X_train: {X_train.shape}")
print(f"Length of X_test: {len(X_test)}   | Shape of X_test: {X_test.shape}")
print(f"Length of y_train: {len(y_train)} | Shape of y_train: {y_train.shape}")
print(f"Length of y_test: {len(y_test)}   | Shape of y_test: {y_test.shape}")

