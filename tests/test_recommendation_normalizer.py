"""
Tests for DIP Signal Normalizer (backend/recommendation/normalizer.py).
"""

import pytest
import pandas as pd
from backend.dataset.dip import generate_dip
from backend.recommendation.normalizer import normalize_dip_signals
from backend.recommendation.schemas import ThresholdConfig


def test_normalizer_extraction_from_dip():
    """Test signal extraction and normalization from a classification DIP dict."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })

    dip_dict = generate_dip(df, target_column="target")
    signals = normalize_dip_signals(dip_dict)

    assert signals.task_type == "classification"
    assert signals.rows == 10
    assert signals.feature_count == 2
    assert signals.dataset_size_category == "small"
    assert signals.missingness_level == "none"
    assert signals.target_missing_flag is False
    assert signals.imbalance_severity == "none"
    assert signals.numerical_heavy_flag is True
    assert signals.categorical_heavy_flag is False


def test_normalizer_missingness_and_imbalance():
    """Test normalizer categorization of missing values and class imbalance."""
    df = pd.DataFrame({
        "cat_feat": ["A", "B", "A", "B", None, "A", "B", "A", "B", "A"],
        "num_feat": [1, None, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    })

    dip_dict = generate_dip(df, target_column="target")
    signals = normalize_dip_signals(dip_dict)

    assert signals.missingness_level in ("low", "moderate")
    assert signals.imbalance_ratio >= 4.0
    assert signals.imbalance_severity == "severe"


def test_normalizer_does_not_mutate_dip():
    """Verify that normalizer does not mutate original DIP dictionary."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]})
    dip_dict = generate_dip(df, target_column="target")
    dip_copy = dict(dip_dict)

    signals = normalize_dip_signals(dip_dict)
    assert dip_dict == dip_copy
