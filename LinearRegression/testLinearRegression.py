import numpy as np
from linearRegression import LinearRegression
from data import X_train, y_train, X_test, y_test


# Calculate mse loss function 
def mse_loss(y_pred, y_true):
    return np.mean((y_true - y_pred) ** 2)

regressor = LinearRegression(lr=0.001, epochs=2000)
regressor.fit(X=X_train, y=y_train)
preds = regressor.predict(X_test)

loss_vals = mse_loss(y_pred=preds, y_true=y_test)
print(loss_vals)