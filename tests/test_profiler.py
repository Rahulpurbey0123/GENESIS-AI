"""Unit tests for profiler and metric calculation modules (DIP v1.1)."""

import pytest
import pandas as pd
import numpy as np
from backend.dataset.profiler import (
    detect_feature_types,
    analyze_missingness,
    analyze_duplicates,
    analyze_target,
    analyze_outliers_iqr,
    analyze_skewness,
    analyze_correlation,
    compute_dataset_profile,
)


def test_feature_type_detection():
    df = pd.DataFrame({
        "num1": [1.0, 2.5, 3.1, 4.2],  # >2 unique values -> numeric
        "cat1": ["a", "b", "c", "d"],  # >2 unique values -> categorical
        "bool1": [True, False, True, False],  # boolean & binary
        "bin_num": [0, 1, 0, 1],  # 2 unique values -> binary
        "bin_str": ["Yes", "No", "Yes", "No"],  # 2 unique values -> binary
        "target": [0, 1, 0, 1]
    })
    res = detect_feature_types(df, feature_cols=["num1", "cat1", "bool1", "bin_num", "bin_str"])
    assert res["numeric_features"] == 1
    assert res["categorical_features"] == 1
    assert res["boolean_features"] == 1
    assert res["binary_features"] == 3  # bool1, bin_num, bin_str
    assert res["numeric_ratio"] == round(1 / 5, 4)
    assert res["binary_ratio"] == round(3 / 5, 4)


def test_binary_feature_detection_types():
    df = pd.DataFrame({
        "bool_col": [True, False, True, False],
        "zero_one_int": [0, 1, 1, 0],
        "two_cat_str": ["Male", "Female", "Male", "Female"],
        "multi_num": [10.0, 20.0, 30.0, 40.0],
        "target": [0, 1, 0, 1]
    })
    res = detect_feature_types(df, feature_cols=["bool_col", "zero_one_int", "two_cat_str", "multi_num"])
    assert "bool_col" in res["binary_columns"]
    assert "zero_one_int" in res["binary_columns"]
    assert "two_cat_str" in res["binary_columns"]
    assert "multi_num" not in res["binary_columns"]
    assert "multi_num" in res["numeric_columns"]


def test_feature_vs_target_missingness_separation():
    # Test missing in features vs target vs both vs none
    df = pd.DataFrame({
        "f1": [1.0, None, 3.0, 4.0],  # 1 missing in features
        "f2": [None, None, 1.0, 2.0],  # 2 missing in features
        "target": [0, 1, None, 1]      # 1 missing in target
    })
    res = analyze_missingness(df, feature_cols=["f1", "f2"], target_column="target")
    
    # Feature missingness (3 missing out of 8 feature cells = 0.375)
    assert res["feature_missingness"]["total_missing"] == 3
    assert res["feature_missingness"]["missing_rate"] == 0.375
    assert res["feature_missingness"]["columns_with_missing"] == 2
    assert res["feature_missingness"]["max_column_missing_rate"] == 0.5  # f2 has 2/4 missing
    
    # Target missingness (1 missing out of 4 target cells = 0.25)
    assert res["target_missingness"]["missing_count"] == 1
    assert res["target_missingness"]["missing_rate"] == 0.25

    # Top-level backward compatibility fields
    assert res["total_missing"] == 3
    assert res["missing_rate"] == 0.375


def test_duplicate_analysis():
    df = pd.DataFrame({
        "f1": [1, 1, 2],
        "target": [0, 0, 1]
    })
    res = analyze_duplicates(df)
    assert res["duplicate_rows"] == 1
    assert res["duplicate_rate"] == round(1 / 3, 4)


def test_target_classification_analysis():
    df = pd.DataFrame({
        "f1": range(10),
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # 8 vs 2 -> imbalance_ratio = 4.0
    })
    res = analyze_target(df, target_column="target")
    assert res["task_type"] == "classification"
    assert res["class_count"] == 2
    assert res["imbalance_ratio"] == 4.0
    assert res["majority_class"] == "0"
    assert res["minority_class"] == "1"


def test_target_regression_analysis():
    targets = np.linspace(100.0, 500.0, 25)
    df = pd.DataFrame({
        "f1": range(25),
        "target": targets
    })
    res = analyze_target(df, target_column="target")
    assert res["task_type"] == "regression"
    assert res["class_count"] is None
    assert res["regression_stats"]["mean"] == round(float(np.mean(targets)), 4)


def test_outlier_detection_iqr():
    data = [10, 12, 11, 13, 12, 11, 10, 12, 11, 100]  # 100 is an outlier
    df = pd.DataFrame({"f1": data, "target": [0]*5 + [1]*5})
    res = analyze_outliers_iqr(df, numeric_cols=["f1"])
    assert res["total_outliers"] == 1
    assert res["columns_with_outliers"] == 1


def test_skewness_analysis():
    df = pd.DataFrame({
        "skewed": [1, 1, 1, 1, 1, 1, 1, 1, 10, 50],
        "target": [0]*5 + [1]*5
    })
    res = analyze_skewness(df, numeric_cols=["skewed"])
    assert res["mean_absolute_skewness"] > 1.0


def test_correlation_analysis():
    x = np.linspace(0, 10, 20)
    y = x * 2.0  # Perfect correlation (r = 1.0)
    df = pd.DataFrame({"x": x, "y": y, "target": [0]*10 + [1]*10})
    res = analyze_correlation(df, numeric_cols=["x", "y"], threshold=0.90)
    assert res["high_correlation_pairs"] == 1
    assert res["max_absolute_correlation"] == 1.0


def test_correlation_edge_cases():
    # Single numeric feature
    df_single = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})
    res_single = analyze_correlation(df_single, numeric_cols=["x"])
    assert res_single["high_correlation_pairs"] == 0
    assert res_single["max_absolute_correlation"] == 0.0

    # Constant feature (variance = 0)
    df_const = pd.DataFrame({"x": [1, 2, 3], "const": [5, 5, 5], "target": [0, 1, 0]})
    res_const = analyze_correlation(df_const, numeric_cols=["x", "const"])
    assert res_const["high_correlation_pairs"] == 0

    # Missing values in numerical features (pairwise complete case)
    df_missing = pd.DataFrame({"x": [1, None, 3, 4], "y": [2, 4, None, 8], "target": [0, 1, 0, 1]})
    res_missing = analyze_correlation(df_missing, numeric_cols=["x", "y"])
    assert isinstance(res_missing["max_absolute_correlation"], float)
