import yaml

from pydantic import BaseModel, Field
from typing import List, Optional


class FeatureConfig(BaseModel):
    """
    Configuration for the feature engineering step.
    """

    numerical: List[str] = Field(
        default_factory=list, description="List of numerical feature names"
    )
    categorical: List[str] = Field(
        default_factory=list, description="List of categorical feature names"
    )


## STEPS CONFIGURATION
class CategoricalStepsConfig(BaseModel):
    """
    Configuration for categorical feature preprocessing steps.
    """

    imputer: Optional[str] = None
    encoder: Optional[str] = None
    imputer_kwargs: Optional[dict] = Field(default_factory=dict)
    encoder_kwargs: Optional[dict] = Field(default_factory=dict)


class NumericalStepsConfig(BaseModel):
    """
    Configuration for numerical feature preprocessing steps.
    """

    imputer: Optional[str] = None
    scaler: Optional[str] = None
    imputer_kwargs: Optional[dict] = Field(default_factory=dict)
    scaler_kwargs: Optional[dict] = Field(default_factory=dict)


class StepsConfig(BaseModel):
    """
    Configuration for the steps in the data preprocessing pipeline.
    """

    categorical: Optional[CategoricalStepsConfig] = None
    numerical: Optional[NumericalStepsConfig] = None


## FEATURES
class PreprocessorConfig(BaseModel):
    """
    Configuration for the data preprocessing step.
    """

    features: FeatureConfig
    steps: StepsConfig


def load_config(config_path: str) -> BaseModel:
    """
    Load the configuration from a YAML file and return it as a Pydantic model.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        BaseModel: Pydantic model containing the configuration.
    """
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    return PreprocessorConfig(**config_data)
