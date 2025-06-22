import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from frigg_ml.src.data_preprocessing import DataPreprocessor
from frigg_ml.src.data_preprocessing import load_config, PreprocessorConfig
from frigg_ml.src.data_loader import DataLoader


def _create_test_data_and_expectations():
    """Create test data and calculate expected values for preprocessing tests."""
    # Create a test DataFrame with categorical and numerical features
    data = pd.DataFrame(
        {
            "Col_0": [1.0, 2.0, 3.0, np.nan, 6.0, 10.0],
            "Col_1": [4.0, 5.0, np.nan, 7.0, 8.0, 9.0],
            "city": ["İstanbul", "Ankara", np.nan, "Ankara", "Antalya", "İzmir"],
        }
    )

    # Calculate expected values manually
    # Expected means for imputation
    col_0_mean = data["Col_0"].mean()  # 4.4
    col_1_mean = data["Col_1"].mean()  # 6.6

    col_0_nan_idx = data["Col_0"].isna().idxmax()
    col_1_nan_idx = data["Col_1"].isna().idxmax()

    # Expected imputed values
    col_0_imputed = data["Col_0"].copy()
    col_1_imputed = data["Col_1"].copy()
    col_0_imputed.iloc[col_0_nan_idx] = col_0_mean
    col_1_imputed.iloc[col_1_nan_idx] = col_1_mean

    # OneHotEncoder typically orders categories alphabetically
    # Expected categories: ['Ankara', 'Antalya', 'İstanbul', 'İzmir']
    expected_ohe = np.array(
        [
            [0, 0, 1, 0],  # İstanbul
            [1, 0, 0, 0],  # Ankara
            [1, 0, 0, 0],  # Ankara (imputed)
            [1, 0, 0, 0],  # Ankara
            [0, 1, 0, 0],  # Antalya
            [0, 0, 0, 1],  # İzmir
        ]
    )

    return data, col_0_imputed, col_1_imputed, expected_ohe, col_0_mean, col_1_mean


def test_data_preprocessor_building(preprocessor, test_datasets_path):
    loader = DataLoader()
    test_path = test_datasets_path / "test.csv"
    data = loader.load_data(test_path)

    assert preprocessor is not None
    assert isinstance(preprocessor, DataPreprocessor)
    assert preprocessor.config is not None
    assert isinstance(preprocessor.config, PreprocessorConfig)


def test_config_based_preprocessor(test_datasets_path):
    """Test that the preprocessor correctly uses the components specified in the config with expected values."""
    # Get test data and expected values
    data, col_0_imputed, col_1_imputed, expected_ohe, col_0_mean, col_1_mean = (
        _create_test_data_and_expectations()
    )

    # Create a custom config with keyword arguments
    config_dict = {
        "features": {"numerical": ["Col_0", "Col_1"], "categorical": ["city"]},
        "steps": {
            "numerical": {
                "imputer": "SimpleImputer",
                "imputer_kwargs": {"strategy": "mean"},
                "scaler": "StandardScaler",
                "scaler_kwargs": {"with_mean": True, "with_std": True},
            },
            "categorical": {
                "imputer": "SimpleImputer",
                "imputer_kwargs": {"strategy": "most_frequent"},
                "encoder": "OneHotEncoder",
                "encoder_kwargs": {"handle_unknown": "ignore", "sparse_output": False},
            },
        },
    }

    config = PreprocessorConfig(**config_dict)

    # Initialize and fit the preprocessor
    preprocessor = DataPreprocessor(config=config)
    transformed_data = preprocessor.fit_transform(data)

    # Verify that the preprocessor was fitted
    assert preprocessor._is_fitted

    # Expected standardized values (mean=0, std=1)
    col_0_standardized = (col_0_imputed - np.mean(col_0_imputed)) / np.std(
        col_0_imputed
    )
    col_1_standardized = (col_1_imputed - np.mean(col_1_imputed)) / np.std(
        col_1_imputed
    )

    # Combine expected numerical and categorical features
    expected_numerical = np.column_stack([col_0_standardized, col_1_standardized])
    expected_full = np.column_stack([expected_numerical, expected_ohe])

    # Verify the transformed data shape
    assert transformed_data.shape == expected_full.shape
    assert transformed_data.shape == (6, 6)  # 2 numerical + 4 one-hot encoded

    # Test the numerical features (first 2 columns)
    np.testing.assert_array_almost_equal(
        transformed_data[:, :2],
        expected_numerical,
        decimal=6,
        err_msg="Numerical features don't match expected values",
    )

    # Test the categorical features (last 4 columns)
    np.testing.assert_array_equal(
        transformed_data[:, 2:],
        expected_ohe,
        err_msg="One-hot encoded features don't match expected values",
    )

    # Test individual expected values for key cases
    # Row 3 (index 3) should have imputed Col_0 value
    expected_col_0_imputed_standardized = (
        col_0_mean - np.mean(col_0_imputed)
    ) / np.std(col_0_imputed)
    np.testing.assert_almost_equal(
        transformed_data[3, 0],
        expected_col_0_imputed_standardized,
        decimal=6,
        err_msg="Imputed Col_0 value doesn't match expected",
    )

    # Row 2 (index 2) should have imputed Col_1 value
    expected_col_1_imputed_standardized = (
        col_1_mean - np.mean(col_1_imputed)
    ) / np.std(col_1_imputed)
    np.testing.assert_almost_equal(
        transformed_data[2, 1],
        expected_col_1_imputed_standardized,
        decimal=6,
        err_msg="Imputed Col_1 value doesn't match expected",
    )

    # Row 2 (index 2) should have imputed city value (Ankara = [1,0,0,0])
    np.testing.assert_array_equal(
        transformed_data[2, 2:],
        [1, 0, 0, 0],
        err_msg="Imputed city value doesn't match expected (should be Ankara)",
    )


def test_config_based_preprocessor_no_scaler(test_datasets_path):
    """Test that the preprocessor works correctly when no scaler is specified in the config."""
    # Get test data and expected values
    data, col_0_imputed, col_1_imputed, expected_ohe, col_0_mean, col_1_mean = (
        _create_test_data_and_expectations()
    )

    # Create a config without scaler
    config_dict = {
        "features": {"numerical": ["Col_0", "Col_1"], "categorical": ["city"]},
        "steps": {
            "numerical": {
                "imputer": "SimpleImputer",
                "imputer_kwargs": {"strategy": "mean"},
                "scaler": None,  # No scaler
            },
            "categorical": {
                "imputer": "SimpleImputer",
                "imputer_kwargs": {"strategy": "most_frequent"},
                "encoder": "OneHotEncoder",
                "encoder_kwargs": {"handle_unknown": "ignore", "sparse_output": False},
            },
        },
    }

    config = PreprocessorConfig(**config_dict)
    preprocessor_no_scaler = DataPreprocessor(config=config)
    transformed_data_no_scaler = preprocessor_no_scaler.fit_transform(data)

    # Verify that the preprocessor was fitted
    assert preprocessor_no_scaler._is_fitted

    # Without scaler, numerical columns should just be the imputed values
    expected_no_scaler = np.column_stack([col_0_imputed, col_1_imputed, expected_ohe])

    # Verify the transformed data shape
    assert transformed_data_no_scaler.shape == expected_no_scaler.shape
    assert transformed_data_no_scaler.shape == (6, 6)  # 2 numerical + 4 one-hot encoded

    np.testing.assert_array_almost_equal(
        transformed_data_no_scaler,
        expected_no_scaler,
        decimal=6,
        err_msg="Transformation without scaler doesn't match expected values",
    )
