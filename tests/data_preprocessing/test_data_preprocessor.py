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
