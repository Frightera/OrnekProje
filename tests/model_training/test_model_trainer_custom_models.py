
import pytest

from frigg_ml.src.model_training.custom_models import BaseCustomModel


class TestCustomModels:
    """Test cases for custom model implementations."""
        
    def test_base_custom_model_abstract(self):
        """Test that BaseCustomModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCustomModel()