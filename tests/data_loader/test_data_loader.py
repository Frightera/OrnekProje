import pytest

from claims_ml.src.data_loader import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader()

def test_check_if_file_extension_supported(data_loader):
    assert data_loader._check_if_file_extension_supported('test.csv') == '.csv'

    with pytest.raises(TypeError):
        data_loader._check_if_file_extension_supported('test.txt')