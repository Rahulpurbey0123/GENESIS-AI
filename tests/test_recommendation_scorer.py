"""
Tests for Recommendation Suitability Scorer (backend/recommendation/scorer.py).
"""

import pytest
from backend.recommendation.registry import PipelineRegistry
from backend.recommendation.schemas import NormalizedDIPSignals, ScoringWeights
from backend.recommendation.scorer import compute_pipeline_score


def test_scorer_bounds_and_determinism():
    """Verify that score is bounded in [0.0, 1.0] and is deterministic across runs."""
    registry = PipelineRegistry()
    pipeline = registry.get_pipeline_by_id("classification_logistic_regression")

    signals = NormalizedDIPSignals(
        task_type="classification",
        rows=500,
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
        dimensionality_ratio=0.008,
        high_dimensional_flag=False,
        dataset_size_category="small",
        categorical_heavy_flag=False,
        numerical_heavy_flag=True,
        complexity_score=1.0,
        complexity_label="Low",
    )

    res1 = compute_pipeline_score(pipeline, signals)
    res2 = compute_pipeline_score(pipeline, signals)

    assert 0.0 <= res1["score"] <= 1.0
    assert res1["score"] == res2["score"]
    assert res1["sub_scores"] == res2["sub_scores"]
    assert res1["reasons"] == res2["reasons"]


def test_scoring_weights_validation():
    """Verify that custom scoring weights are applied properly."""
    registry = PipelineRegistry()
    pipeline = registry.get_pipeline_by_id("classification_logistic_regression")

    signals = NormalizedDIPSignals(
        task_type="classification",
        rows=100,
        columns=3,
        feature_count=2,
        numeric_features=2,
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
        dimensionality_ratio=0.02,
        high_dimensional_flag=False,
        dataset_size_category="small",
        categorical_heavy_flag=False,
        numerical_heavy_flag=True,
        complexity_score=0.5,
        complexity_label="Low",
    )

    custom_weights = ScoringWeights(
        task=0.50,
        dataset_size=0.10,
        feature_type=0.10,
        missingness=0.10,
        imbalance=0.05,
        dimensionality=0.05,
        computational=0.10
    )

    res = compute_pipeline_score(pipeline, signals, weights=custom_weights)
    assert 0.0 <= res["score"] <= 1.0
