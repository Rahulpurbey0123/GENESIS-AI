"""Unit tests for loader module."""

import pytest
from pathlib import Path
import pandas as pd
from backend.dataset.loader import (
    load_csv,
    CSVLoaderError,
    FileNotFoundError,
    InvalidExtensionError,
    EmptyFileError,
    MalformedCSVError,
)

DATA_DIR = Path("data/test_datasets")


def test_valid_csv_filepath():
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    df = load_csv(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "target" in df.columns


def test_missing_file(tmp_path):
    missing_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_csv(missing_path)


def test_invalid_extension(tmp_path):
    txt_path = tmp_path / "dataset.txt"
    txt_path.write_text("a,b,c\n1,2,3")
    with pytest.raises(InvalidExtensionError):
        load_csv(txt_path)


def test_empty_file(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_bytes(b"")
    with pytest.raises(EmptyFileError):
        load_csv(empty_csv)


def test_malformed_csv():
    malformed_bytes = b"header1,header2\nval1\nval1,val2,extra_val\n"
    # Should handle or load as DataFrame without unhandled system crash
    with pytest.raises(CSVLoaderError):
        load_csv(malformed_bytes, filename="bad.csv")


def test_load_bytes():
    valid_bytes = b"col1,col2,target\n1,2,0\n3,4,1\n"
    df = load_csv(valid_bytes, filename="sample.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["col1", "col2", "target"]
