from enum import Enum

SUPPORTED_FILE_EXTENSIONS = [".csv", ".parquet"]


class DataReadingErrorMessages(Enum):
    INVALID_FILE_PATH_TYPE = "Invalid file path type: {type}. Expected str or Path."
    FILE_NOT_FOUND = "Error loading data: File not found at {file_path}."
    EXT_NOT_SUPPORTED = "Error loading data: File extension {ext} is not supported, expected one of {supported_extensions}."
    EMPTY_DATA_FILE = "Error loading data: File is empty."
    PARSER_ERROR = "Error loading data: Error parsing file."
    UNEXPECTED_ERROR = "An unexpected error occurred while loading data: {error}"
