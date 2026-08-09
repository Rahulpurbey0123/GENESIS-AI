"""
Tests for Recommendation Rules Engine (backend/recommendation/rules.py).
"""

import pytest
from backend.recommendation.registry import PipelineRegistry
from backend.recommendation.schemas import NormalizedDIPSignals, ThresholdConfig
from backend.recommendation.rules import evaluate_all_rules


@pytest.fixture
def dummy_signals():
    return NormalizedDIPSignals(
        task_type="classification",
        rows=100000,  # Large dataset
        columns=10,
        feature_count=9,
        numeric_features=9,
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
        imbalance_ratio=5.0,  # Severe imbalance
        imbalance_severity="severe",
        dimensionality_ratio=0.00009,
        high_dimensional_flag=False,
        dataset_size_category="large",
        categorical_heavy_flag=False,
        numerical_heavy_flag=True,
        complexity_score=4.0,
        complexity_label="Medium",
    )


def test_large_dataset_rules_penalize_svc_and_knn(dummy_signals):
    """Verify that rules penalize SVC and KNN on large datasets."""
    registry = PipelineRegistry()
    svc = registry.get_pipeline_by_id("classification_svc")
    hist_gb = registry.get_pipeline_by_id("classification_hist_gradient_boosting")

    svc_res = evaluate_all_rules(svc, dummy_signals)
    hist_res = evaluate_all_rules(hist_gb, dummy_signals)

    assert svc_res.sub_scores["dataset_size"] < 0.50
    assert hist_res.sub_scores["dataset_size"] == 1.0
    assert any("penalty" in r.reason.lower() or "complexity" in r.reason.lower() for r in svc_res.reasons)


def test_imbalance_rules_reward_class_weighted_models(dummy_signals):
    """Verify that imbalanced dataset rules reward models supporting class weights."""
    registry = PipelineRegistry()
    rf = registry.get_pipeline_by_id("classification_random_forest")
    knn = registry.get_pipeline_by_id("classification_k_neighbors")

    rf_res = evaluate_all_rules(rf, dummy_signals)
    knn_res = evaluate_all_rules(knn, dummy_signals)

    assert rf_res.sub_scores["imbalance"] > knn_res.sub_scores["imbalance"]
    assert any("class-weight" in r.reason.lower() for r in rf_res.reasons)
