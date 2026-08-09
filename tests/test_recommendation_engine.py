"""
Integration and End-to-End Tests for RecommendationEngine (backend/recommendation/engine.py).
"""

import pytest
import pandas as pd
from pathlib import Path

from backend.dataset.dip import generate_dip
from backend.recommendation.engine import RecommendationEngine, recommend_pipelines, RecommendationEngineError


DATASETS_DIR = Path(__file__).parent.parent / "data" / "test_datasets"


def test_engine_classification_numerical():
    """Integration test on 01_numerical_classification.csv."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"
    engine = RecommendationEngine()

    report = engine.recommend(csv_path, target_column="target", top_k=5)

    assert report.recommendation_version == "1.1"
    assert report.task_type == "classification"
    assert report.candidate_count_before == 10
    assert report.candidate_count_after_filtering == 5
    assert report.filtering_reduction == 0.50
    assert len(report.recommendations) == 5
    assert report.search_space_reduction == 0.50

    top_rec = report.recommendations[0]
    assert top_rec.rank == 1
    assert 0.0 <= top_rec.score <= 1.0
    assert len(top_rec.reasons) > 0


def test_engine_categorical_heavy():
    """Integration test on 02_categorical_heavy.csv."""
    csv_path = DATASETS_DIR / "02_categorical_heavy.csv"
    engine = RecommendationEngine()

    report = engine.recommend(csv_path, target_column="subscribed", top_k=5)

    assert report.task_type == "classification"
    assert report.candidate_count_after_filtering == 4
    assert len(report.recommendations) == 4
    # HistGradientBoosting or Random Forest should be top candidates for categorical-heavy data
    top_ids = [rec.pipeline_id for rec in report.recommendations[:2]]
    assert any(tid in ("classification_hist_gradient_boosting", "classification_random_forest") for tid in top_ids)


def test_engine_missing_values():
    """Integration test on 03_missing_values.csv."""
    csv_path = DATASETS_DIR / "03_missing_values.csv"
    report_dict = recommend_pipelines(csv_path, target_column="label", top_k=5)

    assert report_dict["task_type"] == "classification"
    assert report_dict["candidate_count_after_filtering"] == 4
    assert len(report_dict["recommendations"]) == 4


def test_engine_imbalanced_classification():
    """Integration test on 04_imbalanced_classification.csv."""
    csv_path = DATASETS_DIR / "04_imbalanced_classification.csv"
    engine = RecommendationEngine()

    report = engine.recommend(csv_path, target_column="is_fraud", top_k=5)

    assert report.task_type == "classification"
    assert len(report.recommendations) == 5

    # Pipelines supporting class weights should rank high
    top_pipeline = report.recommendations[0]
    metadata = top_pipeline.pipeline_metadata
    assert metadata["supports_class_weight"] is True


def test_engine_regression():
    """Integration test on 05_regression.csv."""
    csv_path = DATASETS_DIR / "05_regression.csv"
    engine = RecommendationEngine()

    report = engine.recommend(csv_path, target_column="price", top_k=5)

    assert report.task_type == "regression"
    assert report.candidate_count_before == 10
    assert report.candidate_count_after_filtering == 5
    assert len(report.recommendations) == 5
    assert all(r.pipeline_metadata["task"] == "regression" for r in report.recommendations)


def test_engine_determinism():
    """Verify that calling recommend multiple times produces identical results."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"
    engine = RecommendationEngine()

    r1 = engine.recommend(csv_path, target_column="target", top_k=5)
    r2 = engine.recommend(csv_path, target_column="target", top_k=5)

    assert r1.model_dump() == r2.model_dump()


def test_engine_invalid_inputs():
    """Verify error handling for invalid DIP structures or missing targets."""
    engine = RecommendationEngine()

    with pytest.raises(RecommendationEngineError):
        engine.recommend_from_dip({"invalid": "structure"})

    with pytest.raises(RecommendationEngineError):
        df = pd.DataFrame({"x": [1, 2]})
        engine.recommend(df, target_column=None)
