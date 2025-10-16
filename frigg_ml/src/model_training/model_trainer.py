"""
Model Training Component

This module provides a flexible model training wrapper that follows the scikit-learn API
and supports various machine learning libraries including scikit-learn, XGBoost, and CatBoost.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class ModelTrainer(BaseEstimator):
    """
    A flexible model training wrapper that follows the scikit-learn API.

    This class can accept and train pre-initialized models from various libraries
    while maintaining compatibility with scikit-learn pipelines.

    Parameters
    ----------
    model : Any
        A pre-initialized model instance that has fit() and predict() methods.
        Supports scikit-learn, XGBoost, CatBoost, and custom models.
    model_params : dict, optional
        Additional parameters to set on the model after initialization.

    Attributes
    ----------
    model_ : Any
        The fitted model instance.
    is_fitted_ : bool
        Whether the model has been fitted.
    feature_names_in_ : ndarray of shape (n_features,), optional
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from xgboost import XGBClassifier
    >>> from catboost import CatBoostClassifier
    >>>
    >>> # Using with scikit-learn
    >>> rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> trainer = ModelTrainer(rf_model)
    >>> trainer.fit(X_train, y_train)
    >>> predictions = trainer.predict(X_test)
    >>>
    >>> # Using with XGBoost
    >>> xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    >>> trainer = ModelTrainer(xgb_model)
    >>> trainer.fit(X_train, y_train)
    >>>
    >>> # Using with CatBoost
    >>> cat_model = CatBoostClassifier(iterations=100, random_state=42, verbose=False)
    >>> trainer = ModelTrainer(cat_model)
    >>> trainer.fit(X_train, y_train)
    """

    def __init__(self, model: Any, model_params: Optional[Dict[str, Any]] = None):
        self.model = model
        self.model_params = model_params or {}
        self.is_fitted_ = False

        # Validate that the model has required methods
        self._validate_model()

        # Set additional parameters if provided
        if self.model_params:
            self._set_model_params()

    def _validate_model(self) -> None:
        """Validate that the model has the required fit and predict methods."""
        if not hasattr(self.model, "fit"):
            raise ValueError(
                f"Model {type(self.model).__name__} does not have a 'fit' method. "
                "Ensure your model follows the scikit-learn API."
            )

        if not hasattr(self.model, "predict"):
            raise ValueError(
                f"Model {type(self.model).__name__} does not have a 'predict' method. "
                "Ensure your model follows the scikit-learn API."
            )

    def _set_model_params(self) -> None:
        """Set additional parameters on the model."""
        for param, value in self.model_params.items():
            if hasattr(self.model, param):
                setattr(self.model, param, value)
            else:
                raise ValueError(
                    f"Parameter '{param}' is not valid for model "
                    f"{type(self.model).__name__}"
                )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **fit_params,
    ) -> "ModelTrainer":
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : dict
            Additional parameters to pass to the model's fit method.

        Returns
        -------
        self : ModelTrainer
            Returns self for method chaining.
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        X, y = check_X_y(X, y, accept_sparse=True, allow_nd=True)

        # Store feature information for sklearn compatibility
        self.n_features_in_ = X.shape[1]

        try:
            self.model_ = self.model.fit(X, y, **fit_params)
            self.is_fitted_ = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to fit model {type(self.model).__name__}: {str(e)}"
            ) from e

        return self

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
        check_is_fitted(self, "is_fitted_")

        X = check_array(X, accept_sparse=True, allow_nd=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ModelTrainer is expecting "
                f"{self.n_features_in_} features as seen in fit."
            )

        try:
            return self.model_.predict(X)
        except Exception as e:
            raise RuntimeError(
                f"Failed to predict with model {type(self.model).__name__}: {str(e)}"
            ) from e

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for classification models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        AttributeError
            If the underlying model does not support probability prediction.
        """
        check_is_fitted(self, "is_fitted_")

        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(
                f"Model {type(self.model_).__name__} does not support "
                "probability prediction. Use predict() instead."
            )

        X = check_array(X, accept_sparse=True, allow_nd=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ModelTrainer is expecting "
                f"{self.n_features_in_} features as seen in fit."
            )

        try:
            return self.model_.predict_proba(X)
        except Exception as e:
            raise RuntimeError(
                f"Failed to predict probabilities with model "
                f"{type(self.model_).__name__}: {str(e)}"
            ) from e

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
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
        params = {"model": self.model, "model_params": self.model_params}

        if deep and hasattr(self.model, "get_params"):
            model_params = self.model.get_params(deep=True)
            for key, value in model_params.items():
                params[f"model__{key}"] = value

        return params

    def set_params(self, **params) -> "ModelTrainer":
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : ModelTrainer
            Returns self for method chaining.
        """
        model_params = {}
        trainer_params = {}

        for key, value in params.items():
            if key.startswith("model__"):
                model_param_key = key[7:]  # Remove 'model__' prefix
                model_params[model_param_key] = value
            else:
                trainer_params[key] = value

        for key, value in trainer_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator ModelTrainer")

        if model_params and hasattr(self.model, "set_params"):
            self.model.set_params(**model_params)

        return self

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available from the underlying model.

        Returns
        -------
        feature_importances : ndarray of shape (n_features,) or None
            Feature importances if available, None otherwise.
        """
        check_is_fitted(self, "is_fitted_")

        if hasattr(self.model_, "feature_importances_"):
            return self.model_.feature_importances_
        else:
            return None

    def __repr__(self) -> str:
        """Return string representation of the ModelTrainer."""
        return f"ModelTrainer(model={self.model})"

    def __str__(self) -> str:
        """Return string representation of the ModelTrainer."""
        return self.__repr__()
