import os
from pathlib import Path
import logging
import pandas as pd
from typing import Optional, Union
from .error_messages import DataReadingErrorMessages as EM, SUPPORTED_FILE_EXTENSIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

data_reader_functions = {".csv": pd.read_csv, ".parquet": pd.read_parquet}


class DataLoader:
    """A class for loading data from CSV files"""

    def load_data(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Loads data from the CSV file into a pandas DataFrame.

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the loaded data,
                                     or None if an error occurs.

        Raises:
            TypeError: If the file path is not a string or Path object.
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported or if the file is empty.
        """
        self._validate_file_path(file_path)
        ext = self._check_if_file_extension_supported(file_path)

        reader_func = data_reader_functions.get(ext)
        data: pd.DataFrame = reader_func(file_path)

        if data.empty:
            logger.error(EM.EMPTY_DATA_FILE.value)
            raise ValueError(EM.EMPTY_DATA_FILE.value)

        return data

    def _validate_file_path(self, file_path: Union[str, Path]) -> None:
        """
        Validates the file path.

        Args:
            file_path (Union[str, Path]): The file path to validate.

        Returns:
            bool: True if the file path is valid, False otherwise.
        """
        if not isinstance(file_path, (str, Path)):
            logger.error(EM.INVALID_FILE_PATH_TYPE.value.format(type=type(file_path)))
            raise TypeError(
                EM.INVALID_FILE_PATH_TYPE.value.format(type=type(file_path))
            )

        if not os.path.exists(file_path):
            logger.error(EM.FILE_NOT_FOUND.value.format(file_path=file_path))
            raise FileNotFoundError(EM.FILE_NOT_FOUND.value.format(file_path=file_path))

    def _check_if_file_extension_supported(self, file_path: Union[str, Path]) -> str:
        """
        Checks if the file extension is supported.

        Args:
            file_path (Union[str, Path]): The file path to check.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext = Path(file_path).suffix

        if ext not in SUPPORTED_FILE_EXTENSIONS:
            logger.error(
                EM.EXT_NOT_SUPPORTED.value.format(
                    ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS
                )
            )
            raise ValueError(
                EM.EXT_NOT_SUPPORTED.value.format(
                    ext=ext, supported_extensions=SUPPORTED_FILE_EXTENSIONS
                )
            )

        return ext
