"""
CSV loader module for GENESIS-AI Dataset Intelligence Profile (DIP).

Handles CSV ingestion safely, verifying file existence, extension,
non-emptiness, and structure without modifying data.
"""

from pathlib import Path
import io
from typing import Union, BinaryIO
import pandas as pd


class CSVLoaderError(Exception):
    """Base exception for CSV loading failures."""
    pass


class FileNotFoundError(CSVLoaderError):
    """Raised when the specified file path does not exist."""
    pass


class InvalidExtensionError(CSVLoaderError):
    """Raised when file extension is not .csv."""
    pass


class EmptyFileError(CSVLoaderError):
    """Raised when CSV file is empty (0 bytes or 0 rows)."""
    pass


class MalformedCSVError(CSVLoaderError):
    """Raised when CSV file cannot be parsed or is corrupted."""
    pass


def load_csv(
    file_source: Union[str, Path, BinaryIO, bytes],
    filename: str = "dataset.csv"
) -> pd.DataFrame:
    """
    Safely load a CSV file into a pandas DataFrame.

    Args:
        file_source: Path to CSV file (str or Path), file-like object, or raw bytes.
        filename: Optional name of the file for extension checking when passing bytes/stream.

    Returns:
        pd.DataFrame containing raw parsed dataset.

    Raises:
        CSVLoaderError subclass if file is missing, invalid extension, empty, or malformed.
    """
    if isinstance(file_source, (str, Path)):
        filepath = Path(file_source)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: '{filepath}'")
        if not filepath.is_file():
            raise CSVLoaderError(f"Path is not a regular file: '{filepath}'")
        if filepath.suffix.lower() != ".csv":
            raise InvalidExtensionError(
                f"Only CSV files are supported. Received file with extension: '{filepath.suffix}'"
            )
        
        # Check byte size
        if filepath.stat().st_size == 0:
            raise EmptyFileError(f"Dataset file is empty (0 bytes): '{filepath}'")
        
        try:
            df = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            raise EmptyFileError("Dataset is empty.")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
            raise MalformedCSVError(f"Malformed CSV file could not be parsed: {str(e)}")
        except Exception as e:
            raise CSVLoaderError(f"Unexpected error loading CSV file: {str(e)}")

    elif isinstance(file_source, bytes):
        if filename and not filename.lower().endswith(".csv"):
            raise InvalidExtensionError(
                f"Only CSV files are supported. Received filename: '{filename}'"
            )
        if len(file_source) == 0:
            raise EmptyFileError("Uploaded file is empty (0 bytes).")
        
        try:
            df = pd.read_csv(io.BytesIO(file_source))
        except pd.errors.EmptyDataError:
            raise EmptyFileError("Dataset is empty.")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
            raise MalformedCSVError(f"Malformed CSV file could not be parsed: {str(e)}")
        except Exception as e:
            raise CSVLoaderError(f"Unexpected error loading CSV bytes: {str(e)}")

    elif hasattr(file_source, "read"):
        if filename and not filename.lower().endswith(".csv"):
            raise InvalidExtensionError(
                f"Only CSV files are supported. Received filename: '{filename}'"
            )
        try:
            content = file_source.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
            if len(content) == 0:
                raise EmptyFileError("Uploaded stream is empty (0 bytes).")
            df = pd.read_csv(io.BytesIO(content))
        except pd.errors.EmptyDataError:
            raise EmptyFileError("Dataset is empty.")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
            raise MalformedCSVError(f"Malformed CSV file could not be parsed: {str(e)}")
        except Exception as e:
            raise CSVLoaderError(f"Unexpected error loading CSV stream: {str(e)}")
    else:
        raise CSVLoaderError(f"Unsupported file source type: {type(file_source)}")

    # Check row & column dimensions
    if df.empty or len(df) == 0:
        raise EmptyFileError("Dataset contains 0 rows.")
    if len(df.columns) == 0:
        raise EmptyFileError("Dataset contains 0 columns.")

    return df
