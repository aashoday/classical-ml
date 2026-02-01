import pandas as pd 
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

# Split data in training and testing sets
SPLIT_POINT = int(0.8 * len(X))

X_train, y_train = X[:SPLIT_POINT], y[:SPLIT_POINT]
X_test, y_test = X[SPLIT_POINT:], y[SPLIT_POINT:]

print(f"Shape of X_train {X_train.shape}")
print(f"Shape of X_test {X_test.shape}")
print(f"Shape of y_train {y_train.shape}")
print(f"Shape of y_test {y_test.shape}")