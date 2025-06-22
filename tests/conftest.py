import sys
import os

import pytest
import pandas as pd
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture
def config_path():
    return Path(project_root) / "tests" / "test_datasets" / "test_config.yaml"


@pytest.fixture
def test_datasets_path():
    """Fixture to provide the path to the test datasets."""
    return Path(project_root) / "tests" / "test_datasets"


@pytest.fixture
def empty_dataset(test_datasets_path):
    """Fixture to provide an empty dataset path."""
    return pd.read_csv(test_datasets_path / "test_empty.csv")


@pytest.fixture
def _csv_data(test_datasets_path):
    """Fixture to load CSV data for testing."""
    return pd.read_csv(test_datasets_path / "test.csv")


@pytest.fixture
def _parquet_data(test_datasets_path):
    """Fixture to load Parquet data for testing."""
    return pd.read_parquet(test_datasets_path / "test.parquet")
