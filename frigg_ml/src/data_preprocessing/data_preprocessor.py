import importlib

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from .load_config import PreprocessorConfig
from sklearn.impute import SimpleImputer

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

    def _build_column_transformer(self):
        transformers = []
        
        # Process numerical features
        if self.config.features.numerical:
            for feature in self.config.features.numerical:
                steps = []
                
                # Get imputer class from config
                if self.config.steps.numerical and self.config.steps.numerical.imputer:
                    imputer_class = self._get_class_from_config(self.config.steps.numerical.imputer)
                    # Get kwargs from config or use defaults
                    imputer_kwargs = {}
                    if hasattr(self.config.steps.numerical, 'imputer_kwargs'):
                        imputer_kwargs = self.config.steps.numerical.imputer_kwargs or {}
                    steps.append(("imputer", imputer_class(**imputer_kwargs)))
                        
                # Get scaler class from config
                if self.config.steps.numerical and self.config.steps.numerical.scaler:
                    scaler_class = self._get_class_from_config(self.config.steps.numerical.scaler)
                    # Get kwargs from config or use defaults
                    scaler_kwargs = {}
                    if hasattr(self.config.steps.numerical, 'scaler_kwargs'):
                        scaler_kwargs = self.config.steps.numerical.scaler_kwargs or {}
                    steps.append(("scaler", scaler_class(**scaler_kwargs)))
                
                if steps:  # Only create a pipeline if there are steps
                    numerical_pipeline = Pipeline(steps=steps)
                    transformers.append((feature, numerical_pipeline, [feature]))
        
        # Process categorical features
        if self.config.features.categorical:
            for feature in self.config.features.categorical:
                steps = []
                
                # Get imputer class from config
                if self.config.steps.categorical and self.config.steps.categorical.imputer:
                    imputer_class = self._get_class_from_config(self.config.steps.categorical.imputer)
                    # Get kwargs from config or use defaults
                    imputer_kwargs = {}
                    if hasattr(self.config.steps.categorical, 'imputer_kwargs'):
                        imputer_kwargs = self.config.steps.categorical.imputer_kwargs or {}
                    steps.append(("imputer", imputer_class(**imputer_kwargs)))
                
                # Get encoder class from config
                if self.config.steps.categorical and self.config.steps.categorical.encoder:
                    encoder_class = self._get_class_from_config(self.config.steps.categorical.encoder)
                    # Get kwargs from config or use defaults
                    encoder_kwargs = {}
                    if hasattr(self.config.steps.categorical, 'encoder_kwargs'):
                        encoder_kwargs = self.config.steps.categorical.encoder_kwargs or {}
                    steps.append(("encoder", encoder_class(**encoder_kwargs)))
                
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
