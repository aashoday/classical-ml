from data import *
from logisticRegression import *

regressor = LogisticRegression(lr=0.001, epochs=1000)
regressor.fit(X_train, y_train)
preds = regressor.predict(X_test)
regressor.accuracy(preds, y_test)