import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class LinearRegression:
    def __init__(self, n_iters=1000, lr=0.001):
        self.n_iters = n_iters
        self.lr = lr
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features), dtype=np.float64)
        self.bias = 0.0

        for iter in range(self.n_iters):        
            y_preds = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_preds - y))
            db = (1/n_samples) * np.sum(y_preds - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if(iter % 100 == 0):
            # Print accuracy 
                print(self.maeLoss(y_preds, y))

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return np.dot(X, self.weights) + self.bias

    @staticmethod
    def maeLoss(y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))    

model = LinearRegression(n_iters=2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Test Loss:", model.maeLoss(y_pred, y_test))
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2: {r2_score(y_test, y_pred)}")