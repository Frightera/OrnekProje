import importlib

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from .load_config import PreprocessorConfig


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: PreprocessorConfig):
        self._is_fitted = False
        self.config = config

    def _get_class_from_config(self, class_name, default_class=None):
        """
        Dynamically import the class based on the config name.

        Args:
            class_name (str): Name of the class to import.
            default_class: Default class to use if class_name is None.

        Returns:
            The class object
        """
        if class_name is None:
            return default_class

        # First try to get from already imported modules
        for module_name in ["sklearn.preprocessing", "sklearn.impute"]:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    return getattr(module, class_name)
            except (ImportError, AttributeError):
                pass

        # If not found, try to import directly
        try:
            module_path, class_name = class_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError):
            raise ImportError(f"Could not import {class_name}")

    def _create_step_from_config(
        self, step_name, step_config, config_attr_name, kwargs_attr_name
    ):
        """
        Create a pipeline step from configuration.

        Args:
            step_name (str): Name of the step (e.g., 'imputer', 'scaler', 'encoder')
            step_config: The configuration object containing the step settings
            config_attr_name (str): Name of the attribute containing the class name
            kwargs_attr_name (str): Name of the attribute containing the kwargs

        Returns:
            tuple: (step_name, instantiated_class) or None if step is not configured
        """
        if not step_config or not hasattr(step_config, config_attr_name):
            return None

        class_name = getattr(step_config, config_attr_name)
        if not class_name:
            return None

        step_class = self._get_class_from_config(class_name)

        # Get kwargs from config or use defaults
        step_kwargs = {}
        if hasattr(step_config, kwargs_attr_name):
            step_kwargs = getattr(step_config, kwargs_attr_name) or {}

        return (step_name, step_class(**step_kwargs))

    def _build_column_transformer(self):
        transformers = []

        # Process numerical features
        if self.config.features.numerical:
            for feature in self.config.features.numerical:
                steps = []

                # Add imputer step if configured
                imputer_step = self._create_step_from_config(
                    "imputer", self.config.steps.numerical, "imputer", "imputer_kwargs"
                )
                if imputer_step:
                    steps.append(imputer_step)

                # Add scaler step if configured
                scaler_step = self._create_step_from_config(
                    "scaler", self.config.steps.numerical, "scaler", "scaler_kwargs"
                )
                if scaler_step:
                    steps.append(scaler_step)

                if steps:  # Only create a pipeline if there are steps
                    numerical_pipeline = Pipeline(steps=steps)
                    transformers.append((feature, numerical_pipeline, [feature]))

        # Process categorical features
        if self.config.features.categorical:
            for feature in self.config.features.categorical:
                steps = []

                # Add imputer step if configured
                imputer_step = self._create_step_from_config(
                    "imputer",
                    self.config.steps.categorical,
                    "imputer",
                    "imputer_kwargs",
                )
                if imputer_step:
                    steps.append(imputer_step)

                # Add encoder step if configured
                encoder_step = self._create_step_from_config(
                    "encoder",
                    self.config.steps.categorical,
                    "encoder",
                    "encoder_kwargs",
                )
                if encoder_step:
                    steps.append(encoder_step)

                if steps:  # Only create a pipeline if there are steps
                    categorical_pipeline = Pipeline(steps=steps)
                    transformers.append((feature, categorical_pipeline, [feature]))

        if not transformers:
            raise ValueError("No transformers were created. Check your configuration.")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X, y=None):
        self.pipeline = self._build_column_transformer()
        self.pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError(
                "You must fit the DataPreprocessor before transforming data."
            )
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
