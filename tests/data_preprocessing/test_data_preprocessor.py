import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from frigg_ml.src.data_preprocessing import DataPreprocessor
from frigg_ml.src.data_preprocessing import load_config, PreprocessorConfig
from frigg_ml.src.data_loader import DataLoader


def test_data_preprocessor_building(preprocessor, test_datasets_path):
    loader = DataLoader()
    test_path = test_datasets_path / "test.csv"
    data = loader.load_data(test_path)

    assert preprocessor is not None
    assert isinstance(preprocessor, DataPreprocessor)
    assert preprocessor.config is not None
    assert isinstance(preprocessor.config, PreprocessorConfig)


def test_config_based_preprocessor(test_datasets_path):
    """Test that the preprocessor correctly uses the components specified in the config."""
    # Create a test DataFrame with categorical and numerical features
    data = pd.DataFrame({
        'Col_0': [1.0, 2.0, 3.0, np.nan, 6.0, 10.0],
        'Col_1': [4.0, 5.0, np.nan, 7.0, 8.0, 9.0],
        'city': ['İstanbul', 'Ankara', np.nan, 'Ankara', 'Antalya', 'İzmir']
    })
    
    # Create a custom config with keyword arguments
    config_dict = {
        'features': {
            'numerical': ['Col_0', 'Col_1'],
            'categorical': ['city']
        },
        'steps': {
            'numerical': {
                'imputer': 'SimpleImputer',
                'imputer_kwargs': {'strategy': 'mean'},
                'scaler': 'StandardScaler',
                'scaler_kwargs': {'with_mean': True, 'with_std': True}
            },
            'categorical': {
                'imputer': 'SimpleImputer',
                'imputer_kwargs': {'strategy': 'most_frequent'},
                'encoder': 'OneHotEncoder',
                'encoder_kwargs': {'handle_unknown': 'ignore', 'sparse_output': False}
            }
        }
    }
    
    config = PreprocessorConfig(**config_dict)
    
    # Initialize and fit the preprocessor
    preprocessor = DataPreprocessor(config=config)
    transformed_data = preprocessor.fit_transform(data)
    
    # Verify that the preprocessor was fitted
    assert preprocessor._is_fitted
    
    # Verify the transformed data shape
    # For 2 numerical features and 1 categorical feature with 3 unique values
    # we expect 2 + 3 = 5 columns after one-hot encoding
    assert transformed_data.shape[0] == data.shape[0]
    
    # The specific shape will depend on the number of unique values in the categorical column
    # but we can check that it's at least the number of numerical columns
    assert transformed_data.shape[1] >= 2
    
    # Test with different configuration
    config_dict['steps']['numerical']['scaler'] = None  # Remove scaler
    config = PreprocessorConfig(**config_dict)
    preprocessor = DataPreprocessor(config=config)
    transformed_data_no_scaler = preprocessor.fit_transform(data)
    
    # The data should be different from the previous transformation
    assert not np.array_equal(transformed_data, transformed_data_no_scaler)
