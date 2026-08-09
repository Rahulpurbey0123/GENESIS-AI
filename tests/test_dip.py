"""Integration, non-destructive, determinism, and canonical dataset hash tests for DIP engine v1.1."""

import pytest
from pathlib import Path
import json
import pandas as pd
from backend.dataset.dip import generate_dip, DIPBuilder, compute_dataset_hash
from backend.dataset.loader import load_csv

DATA_DIR = Path("data/test_datasets")


TEST_DATASETS_CONFIG = [
    ("01_numerical_classification.csv", "target", "classification"),
    ("02_categorical_heavy.csv", "subscribed", "classification"),
    ("03_missing_values.csv", "label", "classification"),
    ("04_imbalanced_classification.csv", "is_fraud", "classification"),
    ("05_regression.csv", "price", "regression"),
]


@pytest.mark.parametrize("filename, target_col, expected_task", TEST_DATASETS_CONFIG)
def test_dip_generation_on_5_test_datasets(filename, target_col, expected_task):
    csv_path = DATA_DIR / filename
    assert csv_path.exists(), f"Test dataset file missing: {csv_path}"

    dip = generate_dip(csv_path, target_column=target_col, dataset_name=filename)

    # Validate DIP v1.1 Schema Structure
    assert dip["dip_version"] == "1.1"
    assert isinstance(dip["dataset_hash"], str) and len(dip["dataset_hash"]) == 64
    assert dip["dataset"]["name"] == filename
    assert dip["dataset"]["rows"] > 0
    assert dip["dataset"]["columns"] > 1
    assert dip["dataset"]["feature_count"] == dip["dataset"]["columns"] - 1

    # Schema & Binary Features
    assert "numeric_features" in dip["schema"]
    assert "categorical_features" in dip["schema"]
    assert "binary_features" in dip["schema"]
    assert "binary_ratio" in dip["schema"]

    # Quality & Separated Missingness
    assert "total_missing" in dip["quality"]
    assert "feature_missingness" in dip["quality"]
    assert "target_missingness" in dip["quality"]
    assert "duplicate_rows" in dip["quality"]

    # Statistics & Target
    assert "outlier_rate" in dip["statistics"]
    assert "mean_absolute_skewness" in dip["statistics"]
    assert dip["target"]["name"] == target_col
    assert dip["target"]["task_type"] == expected_task

    # Complexity Score
    score = dip["complexity_score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 10.0
    assert dip["complexity_detail"]["label"] in ["Low", "Medium", "High", "Very High"]

    # Serialization Check
    serialized = json.dumps(dip)
    assert isinstance(serialized, str)


def test_determinism_and_reproducibility():
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    dip1 = generate_dip(csv_path, target_column="target")
    dip2 = generate_dip(csv_path, target_column="target")

    # Exclude non-deterministic runtime measurement field
    dip1_clean = {k: v for k, v in dip1.items() if k != "profiling_time_ms"}
    dip2_clean = {k: v for k, v in dip2.items() if k != "profiling_time_ms"}

    # Hash and output profiles must be identical
    assert dip1["dataset_hash"] == dip2["dataset_hash"]
    assert dip1["complexity_score"] == dip2["complexity_score"]
    assert dip1["statistics"] == dip2["statistics"]
    assert dip1["quality"] == dip2["quality"]
    assert dip1["schema"] == dip2["schema"]
    assert json.dumps(dip1_clean, sort_keys=True) == json.dumps(dip2_clean, sort_keys=True)


def test_hash_determinism():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    h1 = compute_dataset_hash(df)
    h2 = compute_dataset_hash(df)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64


def test_hash_column_order_independence():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    df2 = pd.DataFrame({"c": [5, 6], "a": [1, 2], "b": [3, 4]})
    assert compute_dataset_hash(df1) == compute_dataset_hash(df2)


def test_hash_index_independence():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = df1.copy()
    df2.index = [100, 200, 300]
    assert compute_dataset_hash(df1) == compute_dataset_hash(df2)


def test_hash_data_modification_sensitivity():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = df1.copy()
    df2.iloc[0, 0] = 999  # modify value
    assert compute_dataset_hash(df1) != compute_dataset_hash(df2)


def test_hash_distinct_datasets_uniqueness():
    df1 = load_csv(DATA_DIR / "01_numerical_classification.csv")
    df2 = load_csv(DATA_DIR / "04_imbalanced_classification.csv")
    assert compute_dataset_hash(df1) != compute_dataset_hash(df2)


def test_non_destructive_guarantee():
    original_df = load_csv(DATA_DIR / "03_missing_values.csv")
    cloned_df = original_df.copy(deep=True)

    builder = DIPBuilder()
    dip = builder.build_from_dataframe(cloned_df, target_column="label")

    assert cloned_df.shape == original_df.shape
    assert list(cloned_df.columns) == list(original_df.columns)
    pd.testing.assert_frame_equal(cloned_df, original_df)


def test_custom_dataframe_dip():
    df = load_csv(DATA_DIR / "05_regression.csv")
    builder = DIPBuilder()
    dip = builder.build_from_dataframe(df, target_column="price", dataset_name="custom_housing.csv")
    assert dip["dataset"]["name"] == "custom_housing.csv"
    assert dip["target"]["task_type"] == "regression"
