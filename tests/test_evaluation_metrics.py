"""
Unit Tests for GENESIS-AI Week 7 Evaluation Metrics.
"""

import pytest
import numpy as np
from backend.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_efficiency_metrics,
    get_primary_metric_name,
    is_higher_better,
    is_better_score
)


def test_classification_metrics():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    res = calculate_classification_metrics(y_true, y_pred)
    assert "f1" in res
    assert "accuracy" in res
    assert 0.0 <= res["f1"] <= 1.0
    assert res["accuracy"] == 0.80


def test_regression_metrics():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    res = calculate_regression_metrics(y_true, y_pred)
    assert "rmse" in res
    assert "mae" in res
    assert "r2" in res
    assert res["rmse"] > 0.0


def test_primary_metric_and_direction():
    assert get_primary_metric_name("classification") == "f1"
    assert get_primary_metric_name("regression") == "rmse"

    assert is_higher_better("f1") is True
    assert is_higher_better("rmse") is False

    assert is_better_score(0.9, 0.8, "f1") is True
    assert is_better_score(10.0, 15.0, "rmse") is True


def test_efficiency_metrics():
    res = calculate_efficiency_metrics(
        candidate_count_before=10,
        candidate_count_after=4,
        evaluations_used=45,
        runtime_seconds=3.21
    )
    assert res["candidate_space_reduction"] == 0.60
    assert res["evaluations_used"] == 45
    assert res["runtime_seconds"] == 3.21
