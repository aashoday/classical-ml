import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)

TEST_SIZE = int(0.2*len(X))

X_train, y_train = X[:TEST_SIZE], y[:TEST_SIZE]
X_test, y_test = X[TEST_SIZE:], y[:TEST_SIZE:]

print(f"Sizes X train: {X_train.shape} y_train: {y_train.shape}, X test: {X_test.shape}, y_test: {y_test.shape}")

class LinearRegression:
    def __init__(self, n_iters=1000, lr=0.001):
        self.n_iters = n_iters
        self.lr = lr
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...