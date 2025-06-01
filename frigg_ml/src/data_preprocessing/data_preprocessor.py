from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from .load_config import PreprocessorConfig


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: PreprocessorConfig):
        self._is_fitted = False
        self.config = config

    def _build_column_transformer(self):
        transformers = []
        for feature in self.config.features.numerical:
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append((feature, numerical_pipeline, [feature]))

        for feature in self.config.features.categorical:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),  # TODO: Model yazımında değinilip, değiştirilecek.
                ]
            )
            transformers.append((feature, categorical_pipeline, [feature]))
        # TODO: Buraya feature sayısını check eden bir kod eklenebilir.
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
