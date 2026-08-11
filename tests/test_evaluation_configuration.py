"""
Unit Tests for GENESIS-AI Week 7 Evaluation Configuration.
"""

import pytest
from backend.evaluation.configuration import get_default_benchmark_config, METHOD_METADATA, DEFAULT_DATASET_SPECS


def test_default_config_construction():
    config = get_default_benchmark_config()
    assert config.benchmark_version == "1.0"
    assert len(config.datasets) == 5
    assert len(config.methods) == 5
    assert config.seeds == [42, 123, 456, 789, 2024]


def test_method_metadata():
    assert "method_a_full_genesis" in METHOD_METADATA
    assert "method_e_unguided_baseline" in METHOD_METADATA

    meta_a = METHOD_METADATA["method_a_full_genesis"]
    assert meta_a["code"] == "A"
    assert meta_a["has_dip"] is True
    assert meta_a["has_recommendation"] is True
    assert meta_a["has_optimization"] is True

    meta_e = METHOD_METADATA["method_e_unguided_baseline"]
    assert meta_e["code"] == "E"
    assert meta_e["has_dip"] is False
    assert meta_e["has_optimization"] is False


def test_default_dataset_specs():
    assert len(DEFAULT_DATASET_SPECS) == 5
    filenames = [s.filename for s in DEFAULT_DATASET_SPECS]
    assert "01_numerical_classification.csv" in filenames
    assert "05_regression.csv" in filenames
