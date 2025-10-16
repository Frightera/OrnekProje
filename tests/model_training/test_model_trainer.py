"""
Tests for the model training component.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from frigg_ml.src.model_training import ModelTrainer


class TestModelTrainerInit:
    def test_init_with_valid_model(self):
        """Test initialization with a valid model."""
        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)

        assert trainer.model == model
        assert trainer.model_params == {}
        assert not trainer.is_fitted_

    def test_init_with_model_params(self):
        """Test initialization with additional model parameters."""
        model = RandomForestClassifier(random_state=42)
        params = {"n_estimators": 200, "max_depth": 10}
        trainer = ModelTrainer(model, model_params=params)

        assert trainer.model.n_estimators == 200
        assert trainer.model.max_depth == 10
        assert trainer.model_params == params

    def test_init_with_invalid_model(self):
        """Test initialization with a model that doesn't have required methods."""
        invalid_model = Mock()
        delattr(invalid_model, "fit")

        with pytest.raises(ValueError, match="does not have a 'fit' method"):
            ModelTrainer(invalid_model)

    def test_init_with_invalid_model_params(self):
        """Test initialization with invalid model parameters."""
        model = RandomForestClassifier(random_state=42)
        params = {"invalid_param": 100}

        with pytest.raises(ValueError, match="Parameter 'invalid_param' is not valid"):
            ModelTrainer(model, model_params=params)


class TestModelTrainer:
    """Test cases for the ModelTrainer class."""

    def test_fit_classification(self, classification_data):
        """Test fitting with classification data."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)

        # Test fitting
        result = trainer.fit(X_train, y_train)

        # Check return value is self
        assert result is trainer
        assert trainer.is_fitted_
        assert hasattr(trainer, "model_")
        assert trainer.n_features_in_ == X_train.shape[1]

    def test_fit_regression(self, regression_data):
        """Test fitting with regression data."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomForestRegressor(random_state=42)
        trainer = ModelTrainer(model)

        # Test fitting
        result = trainer.fit(X_train, y_train)

        # Check return value is self
        assert result is trainer
        assert trainer.is_fitted_
        assert hasattr(trainer, "model_")
        assert trainer.n_features_in_ == X_train.shape[1]

    def test_fit_with_pandas(self, classification_data):
        """Test fitting with pandas DataFrame and Series."""
        X_train, X_test, y_train, y_test = classification_data

        # Convert to pandas
        X_train_df = pd.DataFrame(
            X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])]
        )
        y_train_series = pd.Series(y_train, name="target")

        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)

        # Test fitting
        trainer.fit(X_train_df, y_train_series)

        assert trainer.is_fitted_
        assert hasattr(trainer, "feature_names_in_")
        assert len(trainer.feature_names_in_) == X_train.shape[1]

    def test_predict_classification(self, classification_data):
        """Test prediction with classification model."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)
        trainer.fit(X_train, y_train)

        # Test prediction
        predictions = trainer.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_test),)
        assert set(predictions).issubset(set(y_train))

    def test_predict_regression(self, regression_data):
        """Test prediction with regression model."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomForestRegressor(random_state=42)
        trainer = ModelTrainer(model)
        trainer.fit(X_train, y_train)

        # Test prediction
        predictions = trainer.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_test),)

    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)
        trainer.fit(X_train, y_train)

        # Test probability prediction
        probabilities = trainer.predict_proba(X_test)

        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_test), 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_proba_without_support(self, regression_data):
        """Test probability prediction with model that doesn't support it."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomForestRegressor(random_state=42)
        trainer = ModelTrainer(model)
        trainer.fit(X_train, y_train)

        # Test that predict_proba raises appropriate error
        with pytest.raises(
            AttributeError, match="does not support probability prediction"
        ):
            trainer.predict_proba(X_test)

    def test_predict_before_fit(self, classification_data):
        """Test prediction before fitting raises error."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)

        # Should raise error since model is not fitted
        with pytest.raises(Exception):  # sklearn raises NotFittedError
            trainer.predict(X_test)

    def test_feature_importances(self, classification_data):
        """Test feature importances property."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)
        trainer.fit(X_train, y_train)

        # Test feature importances
        importances = trainer.feature_importances_

        assert importances is not None
        assert isinstance(importances, np.ndarray)
        assert len(importances) == X_train.shape[1]

    def test_feature_importances_without_support(self, classification_data):
        """Test feature importances with model that doesn't support it."""
        X_train, X_test, y_train, y_test = classification_data

        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        trainer.fit(X_train, y_train)

        # LogisticRegression doesn't have feature_importances_
        importances = trainer.feature_importances_
        assert importances is None

    def test_get_params(self):
        """Test get_params method."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        params = {"max_depth": 10}
        trainer = ModelTrainer(model, model_params=params)

        all_params = trainer.get_params(deep=True)

        assert "model" in all_params
        assert "model_params" in all_params
        assert "model__n_estimators" in all_params
        assert all_params["model__n_estimators"] == 100

    def test_set_params(self):
        """Test set_params method."""
        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)

        # Set parameters
        trainer.set_params(model__n_estimators=200, model__max_depth=10)

        assert trainer.model.n_estimators == 200
        assert trainer.model.max_depth == 10
