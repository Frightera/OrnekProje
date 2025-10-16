Users can create their custom models as follows (if they are compatible with Sklearn):

```python
class LinearRegressionExample(SklearnCompatibleModel):
    """
    Example implementation of a simple linear regression model.
    
    This serves as a template for creating custom models.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights_ = None
        self.bias_ = None
        self.is_fitted_ = False
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> 'LinearRegressionExample':
        """Fit the linear regression model using gradient descent."""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0
        
        # Gradient descent
        for _ in range(self.max_iterations):
            # Forward pass
            y_pred = X.dot(self.weights_) + self.bias_
            
            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update weights
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return X.dot(self.weights_) + self.bias_
```

This follows the Scikit-Learn's Public API.

It's also possible to do this in classification:
```python
class LogisticRegressionExample(SklearnCompatibleModel):
    """
    Example implementation of a simple logistic regression model.
    
    This serves as a template for creating custom classification models.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights_ = None
        self.bias_ = None
        self.is_fitted_ = False
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> 'LogisticRegressionExample':
        """Fit the logistic regression model using gradient descent."""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0
        
        # Gradient descent
        for _ in range(self.max_iterations):
            # Forward pass
            linear_pred = X.dot(self.weights_) + self.bias_
            y_pred = self._sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update weights
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
            
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        linear_pred = X.dot(self.weights_) + self.bias_
        probabilities = self._sigmoid(linear_pred)
        
        # Return probabilities for both classes
        return np.column_stack([1 - probabilities, probabilities])
```

In other words, you need to:
1) Define `fit()` and `predict()`methods.

Please see the ABC definition in `custom_models.py`