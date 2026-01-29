# Import Libraries 
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Fetch data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Set random seed 
RANDOM_SEED = 911

# Split data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_SEED)

# Print length and shape of splits 
print(f"Length of X_train: {len(X_train)} | Shape of X_train: {X_train.shape}")
print(f"Length of X_test: {len(X_test)} | Shape of X_test: {X_test.shape}")
print(f"Length of y_train: {len(y_train)} | Shape of y_train: {y_train.shape}")
print(f"Length of y_test: {len(y_test)} | Shape of y_test: {y_test.shape}")

