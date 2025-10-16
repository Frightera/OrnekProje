"""
Model Training Module

This module provides a flexible model training component that follows the
scikit-learn API and supports various machine learning libraries.
"""

from .model_trainer import ModelTrainer
from .custom_models import (
    BaseCustomModel,
    SklearnCompatibleModel,
)

__all__ = [
    "ModelTrainer",
    "BaseCustomModel",
    "SklearnCompatibleModel",
]
