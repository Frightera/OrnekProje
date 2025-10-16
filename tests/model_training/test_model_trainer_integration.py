import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from frigg_ml.src.model_training.model_trainer import ModelTrainer
from tests.model_training.test_utils import MockCustomModel


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_mock_custom_model_with_trainer(self):
        """Test MockCustomModel with ModelTrainer."""
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        model = MockCustomModel()
        trainer = ModelTrainer(model)

        # Fit and predict
        trainer.fit(X, y)
        predictions = trainer.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert trainer.is_fitted_

    def test_pipeline_integration(self, classification_data):
        """Test integration with scikit-learn pipeline."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomForestClassifier(random_state=42)
        trainer = ModelTrainer(model)

        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", trainer)])

        assert isinstance(pipeline.steps[-1][-1], ModelTrainer), (
            f"Expected ModelTrainer in pipeline, got {type(pipeline.steps[-1][-1])}"
        )

        # Fit and predict
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X_test),)
