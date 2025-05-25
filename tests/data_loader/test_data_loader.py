import pytest
import re
import pandas as pd
from frigg_ml.src.data_loader import DataLoader


@pytest.fixture
def data_loader():
    return DataLoader()


# Expected Value Tests
@pytest.mark.parametrize(
    "file_name, expected_data",
    [
        pytest.param("test.csv", "_csv_data", id="csv"),
        pytest.param("test.parquet", "_parquet_data", id="parquet"),
    ],
)
def test_load_data_with_valid_file(
    data_loader, test_datasets_path, request, file_name, expected_data
):
    file_path = test_datasets_path / file_name
    expected_df = request.getfixturevalue(expected_data)
    
    data = data_loader.load_data(file_path)
    assert isinstance(data, pd.DataFrame)
    pd.testing.assert_frame_equal(data, expected_df)


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("test_empty.csv", id="empty_csv"),
    ],
)
def test_load_data_with_empty_file(
    data_loader, test_datasets_path, file_name
):
    file_path = test_datasets_path / file_name
    
    with pytest.raises(ValueError, match="Error loading data: File is empty."):
        data_loader.load_data(file_path)


# Method Test - 1 - Expected Value
@pytest.mark.parametrize(
    "extension, file_name",
    [
        (".csv", "test.csv"),
        (".parquet", "test.parquet"),
        (".csv", "test.parquet.csv"),
        (".parquet", "test.csv.parquet"),
        (".csv", "test/file.csv"),
        (".parquet", "test/file.parquet"),
    ],
)
def test_check_if_file_extension_returns_correct_result(
    data_loader, extension, file_name
):
    assert data_loader._check_if_file_extension_supported(file_name) == extension


# Method Test - 1 - Expected Error Messages
@pytest.mark.parametrize(
    "file_ext",
    [
        ".txt",
        ".xlsx",
        ".json",
        ".xml",
    ],
)
def test_check_if_file_extension_throws_value_error_when_unsupported(
    data_loader, file_ext
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Error loading data: File extension {file_ext} is not supported, expected one of ['.csv', '.parquet']."
        ),
    ):
        data_loader._check_if_file_extension_supported(f"test{file_ext}")


@pytest.mark.parametrize(
    "invalid_path_input, error_type, error_message_match_template",
    [
        pytest.param(
            12345,
            TypeError,
            "Invalid file path type: <class 'int'>. Expected str or Path.",
            id="invalid_type",
        ),
        pytest.param(
            "non_existent_file.csv",
            FileNotFoundError,
            "Error loading data: File not found at {path}.",
            id="file_not_found",
        ),
    ],
)
def test_validate_file_path_raises_errors(
    data_loader,
    test_datasets_path,
    invalid_path_input,
    error_type,
    error_message_match_template,
):
    path_to_check = invalid_path_input
    if (
        isinstance(invalid_path_input, str)
        and "non_existent_file" in invalid_path_input
    ):
        path_to_check = test_datasets_path / invalid_path_input

    expected_message = error_message_match_template
    if "{path}" in error_message_match_template:
        expected_message = error_message_match_template.format(path=path_to_check)

    with pytest.raises(error_type, match=re.escape(expected_message)):
        data_loader._validate_file_path(path_to_check)
