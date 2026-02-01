# Import libraries
import numpy as np

# Create linearRegression model class
class LinearRegression:
    """
    Linear Regression model trained using Gradient Descent.

    This implementation supports feature standardization, mean absolute error (mae) monitoring, and vectorised gradient updates.
    """

    # Initialize parameters
    def __init__(self, lr: float = 0.001, epochs: int = 1000) -> None:
        """
        Initialize the Linear Regression model.

        Parameters
        ----------
        lr : float, default=0.001 --Learning rate for gradient descent.
        epochs : int, default=1000 --Number of training iterations.
        """
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.X_mean = None
        self.X_std = None

    # Fit model to data
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model to training data.
        This method standardizes features, initializes parameters, and performs gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) --Training input samples. 
        y : np.ndarray of shape (n_samples,) --Target Values.
        
        """
        n_samples , n_features = X.shape

        # Scale features
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X = (X - self.X_mean) / self.X_std
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for epoch in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Print metrics
            log_interval = max(1, self.epochs // 10)
            if epoch % log_interval == 0 or epoch == self.epochs -1:
                print(f"Epoch: {epoch}")
                print(f"MAE Loss: {self.maeLoss(y_predicted, y)}")

    # Make prediction on test cases
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using trained model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) --Input samples.
        
        Returns 
        -------
        np.ndarray of shape (n_samples,) --Predicted target values.
        
        """
        X = (X - self.X_mean) / self.X_std
        return np.dot(X, self.weights) + self.bias
    
    # Calculate loss
    @staticmethod
    def maeLoss(preds: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Absolute Error (MAE)

        Parameters
        ----------
        preds : np.ndarray of shape (n_samples,) --Predicted target values.
        y : np.ndarray of shape (n_samples,) --True target values.

        Returns
        -------
        float --Mean Absolute Error.
        """
        return np.mean(np.abs(preds - y))