"""Unit tests for validator module."""

import pytest
import pandas as pd
from backend.dataset.validator import validate_dataset, DatasetValidationError


def test_valid_dataset_and_target():
    df = pd.DataFrame({
        "age": [25, 30, 45, 50],
        "income": [50000, 60000, 80000, 90000],
        "target": [0, 1, 0, 1]
    })
    res = validate_dataset(df, target_column="target")
    assert res["is_valid"] is True
    assert res["target_column"] == "target"
    assert res["feature_count"] == 2


def test_missing_target_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(df, target_column="non_existent")
    assert "Target column 'non_existent' was not found." in str(exc_info.value)


def test_empty_dataset():
    df = pd.DataFrame()
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(df, target_column="target")
    assert "Dataset is empty." in str(exc_info.value)


def test_one_class_target():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "target": [1, 1, 1, 1]  # Single unique value
    })
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(df, target_column="target")
    assert "insufficient unique values" in str(exc_info.value)


def test_insufficient_columns():
    df = pd.DataFrame({"target": [0, 1, 0, 1]})
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(df, target_column="target")
    assert "at least 2 columns" in str(exc_info.value)
