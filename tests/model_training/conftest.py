import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def classification_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)
