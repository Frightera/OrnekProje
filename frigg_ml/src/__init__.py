"""
Frigg ML - Machine Learning Components
"""

from .model_training import (
    ModelTrainer,
    BaseCustomModel,
    SklearnCompatibleModel,
)

__all__ = [
    "ModelTrainer",
    "BaseCustomModel",
    "SklearnCompatibleModel", 
]