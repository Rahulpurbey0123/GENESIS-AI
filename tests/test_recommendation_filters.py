"""
Tests for Stage 1 Compatibility Filters (backend/recommendation/filters.py).
"""

import pytest
from backend.recommendation.registry import PipelineRegistry
from backend.recommendation.schemas import NormalizedDIPSignals
from backend.recommendation.filters import (
    filter_task_compatibility,
    apply_compatibility_filters,
)


@pytest.fixture
def base_signals():
    return NormalizedDIPSignals(
        task_type="classification",
        rows=100,
        columns=5,
        feature_count=4,
        numeric_features=4,
        categorical_features=0,
        binary_features=0,
        numeric_ratio=1.0,
        categorical_ratio=0.0,
        binary_ratio=0.0,
        total_missing=0,
        missing_rate=0.0,
        feature_missingness=0.0,
        target_missingness=0.0,
        missingness_level="none",
        target_missing_flag=False,
        outlier_rate=0.0,
        mean_absolute_skewness=0.0,
        imbalance_ratio=1.0,
        imbalance_severity="none",
        dimensionality_ratio=0.04,
        high_dimensional_flag=False,
        dataset_size_category="small",
        categorical_heavy_flag=False,
        numerical_heavy_flag=True,
        complexity_score=1.0,
        complexity_label="Low",
    )


def test_filter_task_compatibility_classification(base_signals):
    """Test that task filter keeps only classification pipelines for classification task."""
    registry = PipelineRegistry()
    all_candidates = registry.get_all_pipelines()
    filtered = filter_task_compatibility(all_candidates, base_signals)

    assert len(filtered) == 5
    assert all(p.task == "classification" for p in filtered)


def test_filter_task_compatibility_regression(base_signals):
    """Test that task filter keeps only regression pipelines for regression task."""
    registry = PipelineRegistry()
    all_candidates = registry.get_all_pipelines()

    reg_signals = base_signals.model_copy(update={"task_type": "regression"})
    filtered = filter_task_compatibility(all_candidates, reg_signals)

    assert len(filtered) == 5
    assert all(p.task == "regression" for p in filtered)


def test_apply_compatibility_filters_warnings(base_signals):
    """Verify warning generation when target missingness is present."""
    registry = PipelineRegistry()
    all_candidates = registry.get_all_pipelines()

    missing_target_signals = base_signals.model_copy(update={"target_missing_flag": True, "target_missingness": 0.05})
    filtered, warnings = apply_compatibility_filters(all_candidates, missing_target_signals)

    assert len(filtered) == 5
    assert any("Target column contains missing values" in w for w in warnings)
