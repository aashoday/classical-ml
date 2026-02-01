import numpy as np

class LogisticRegression:
    def __init__(self, lr: float = 0.001, epochs: int = 1000) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_vals = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_vals)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_vals = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_vals)
        prediction_cls = [1 if i > 0.5 else 0 for i in predictions]
        return prediction_cls

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def accuracy(preds: np.ndarray , y: np.ndarray) -> float:
        print(f"Accuracy : {np.sum(preds == y) / len(y)}")
