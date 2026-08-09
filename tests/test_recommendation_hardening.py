"""
Hardening and Correction Regression Tests for Recommendation Engine v1.1 (tests/test_recommendation_hardening.py).
"""

import math
import pytest
from pathlib import Path
from pydantic import ValidationError

from backend.recommendation.schemas import (
    ScoringWeights,
    RecommendationReason,
    RecommendationReport
)
from backend.recommendation.registry import PipelineRegistry
from backend.recommendation.rules import evaluate_all_rules, RuleEvaluationResult
from backend.recommendation.normalizer import normalize_dip_signals
from backend.recommendation.engine import RecommendationEngine

DATASETS_DIR = Path(__file__).parent.parent / "data" / "test_datasets"


def test_filtering_reduction_independent_of_top_k():
    """Fix #1: Verify filtering_reduction measures hard filtering (10->5 = 50%) and is unaffected by top_k."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"
    engine = RecommendationEngine()

    report_k3 = engine.recommend(csv_path, target_column="target", top_k=3)
    report_k5 = engine.recommend(csv_path, target_column="target", top_k=5)

    # 10 total candidates before -> 5 classification candidates after filtering
    assert report_k3.candidate_count_before == 10
    assert report_k3.candidate_count_after_filtering == 5
    assert report_k3.filtering_reduction == 0.50
    assert report_k3.recommended_count == 3
    assert report_k3.top_k_selection_ratio == 0.60

    # Top-K = 5 should NOT alter filtering_reduction
    assert report_k5.candidate_count_before == 10
    assert report_k5.candidate_count_after_filtering == 5
    assert report_k5.filtering_reduction == 0.50
    assert report_k5.recommended_count == 5
    assert report_k5.top_k_selection_ratio == 1.0


def test_scoring_weights_validation_valid():
    """Fix #2: Verify valid weights summing to 1.0 (exact and within tolerance) are accepted."""
    # Default weights sum to 1.0
    w_default = ScoringWeights()
    assert abs(sum([w_default.task, w_default.dataset_size, w_default.feature_type, w_default.missingness, w_default.imbalance, w_default.dimensionality, w_default.computational]) - 1.0) <= 1e-6

    # Valid floating point sum within 1e-6 tolerance
    w_tol = ScoringWeights(
        task=0.2000001,
        dataset_size=0.15,
        feature_type=0.20,
        missingness=0.10,
        imbalance=0.10,
        dimensionality=0.10,
        computational=0.15
    )
    assert w_tol is not None


def test_scoring_weights_validation_invalid_sums():
    """Fix #2: Verify rejection of invalid weight sums without silent auto-normalization."""
    # Sum below 1.0 (0.95)
    with pytest.raises(ValidationError, match="Recommendation scoring weights must sum to 1.0"):
        ScoringWeights(
            task=0.15, dataset_size=0.15, feature_type=0.20,
            missingness=0.10, imbalance=0.10, dimensionality=0.10, computational=0.15
        )

    # Sum above 1.0 (1.05)
    with pytest.raises(ValidationError, match="Recommendation scoring weights must sum to 1.0"):
        ScoringWeights(
            task=0.25, dataset_size=0.15, feature_type=0.20,
            missingness=0.10, imbalance=0.10, dimensionality=0.10, computational=0.15
        )

    # Large invalid sum (3.5)
    with pytest.raises(ValidationError, match="Recommendation scoring weights must sum to 1.0"):
        ScoringWeights(
            task=0.50, dataset_size=0.50, feature_type=0.50,
            missingness=0.50, imbalance=0.50, dimensionality=0.50, computational=1.0
        )


def test_scoring_weights_validation_invalid_values():
    """Fix #2: Verify rejection of negative, NaN, or infinite weights."""
    with pytest.raises(ValidationError):
        ScoringWeights(task=-0.10, dataset_size=0.25, feature_type=0.20, missingness=0.10, imbalance=0.10, dimensionality=0.20, computational=0.25)

    with pytest.raises(ValidationError):
        ScoringWeights(task=float("nan"), dataset_size=0.15, feature_type=0.20, missingness=0.10, imbalance=0.10, dimensionality=0.10, computational=0.15)

    with pytest.raises(ValidationError):
        ScoringWeights(task=float("inf"), dataset_size=0.15, feature_type=0.20, missingness=0.10, imbalance=0.10, dimensionality=0.10, computational=0.15)


def test_hist_gradient_boosting_categorical_metadata():
    """Fix #3: Verify HistGradientBoosting entries set handles_categorical_natively = False due to OrdinalEncoder step."""
    registry = PipelineRegistry()
    clf_hgb = registry.get_pipeline_by_id("classification_hist_gradient_boosting")
    reg_hgb = registry.get_pipeline_by_id("regression_hist_gradient_boosting")

    assert clf_hgb is not None
    assert clf_hgb.handles_categorical_natively is False

    assert reg_hgb is not None
    assert reg_hgb.handles_categorical_natively is False


def test_rule_traceability_and_heuristic_reasons():
    """Fix #4: Verify rule reasons use RecommendationReason objects with rule_id and heuristic wording."""
    registry = PipelineRegistry()
    pipeline = registry.get_pipeline_by_id("classification_logistic_regression")

    csv_path = DATASETS_DIR / "01_numerical_classification.csv"
    engine = RecommendationEngine()
    report = engine.recommend(csv_path, target_column="target", top_k=3)

    assert len(report.recommendations) > 0
    top_rec = report.recommendations[0]

    assert len(top_rec.reasons) > 0
    for reason_obj in top_rec.reasons:
        assert isinstance(reason_obj, RecommendationReason)
        assert reason_obj.rule_id.startswith("RULE_")
        assert len(reason_obj.reason) > 0
        # Ensure no claims of guaranteed best predictive performance
        assert "definitely perform best" not in reason_obj.reason.lower()
        assert "guaranteed" not in reason_obj.reason.lower()
