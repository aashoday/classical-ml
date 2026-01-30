import numpy as np
from linearRegression import LinearRegression
from data import X_train, y_train, X_test, y_test

regressor = LinearRegression(lr=0.001, epochs=2000)
regressor.fit(X=X_train, y=y_train)
preds = regressor.predict(X_test)

loss_vals = regressor.maeLoss(preds=preds, y=y_test)
print(f"\nTesting loss : {loss_vals}")