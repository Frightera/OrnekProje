"""
Base classes and interfaces for custom models.

This module provides base classes and clear patterns for users to create
their own custom model classes while maintaining compatibility with the
ModelTrainer component.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd


class BaseCustomModel(ABC):
    """
    Abstract base class for custom models.

    This class defines the interface that custom models must implement
    to be compatible with the ModelTrainer component.

    All custom models should inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "BaseCustomModel":
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional fitting parameters.

        Returns
        -------
        self : BaseCustomModel
            Returns self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        pass


class SklearnCompatibleModel(BaseCustomModel):
    """
    A mixin class that provides sklearn-compatible methods.

    This class can be used as a base for custom models that want
    to be fully compatible with scikit-learn pipelines and utilities.
    """

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        # Get all non-private attributes that don't end with '_'
        params = {}
        for key in dir(self):
            if not key.startswith("_") and not key.endswith("_"):
                value = getattr(self, key)
                # Only include simple types and avoid methods
                if not callable(value):
                    params[key] = value
        return params

    def set_params(self, **params) -> "SklearnCompatibleModel":
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : SklearnCompatibleModel
            Returns self for method chaining.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {type(self).__name__}"
                )
        return self
