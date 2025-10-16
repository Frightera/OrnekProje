from frigg_ml.src.model_training.custom_models import BaseCustomModel
import numpy as np


class MockCustomModel(BaseCustomModel):
    """Mock implementation of BaseCustomModel for testing."""

    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y, **kwargs):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Model must be fitted")
        return np.zeros(len(X))
