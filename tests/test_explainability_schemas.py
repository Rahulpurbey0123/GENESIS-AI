"""
Unit tests for backend/explainability/schemas.py.
"""

import pytest
from backend.explainability.schemas import (
    FeatureImportanceRecord,
    FeatureContribution,
    LocalExplanationRecord,
    ExplanationOutput
)


def test_feature_importance_record():
    rec = FeatureImportanceRecord(feature="age", importance=0.45, rank=1, direction=1)
    assert rec.feature == "age"
    assert rec.importance == 0.45
    assert rec.rank == 1
    assert rec.direction == 1
    assert rec.mean_importance is None


def test_feature_contribution():
    fc = FeatureContribution(feature="income", feature_value=50000.0, contribution=1.23)
    assert fc.feature == "income"
    assert fc.feature_value == 50000.0
    assert fc.contribution == 1.23


def test_local_explanation_record():
    fc = FeatureContribution(feature="x1", feature_value=2.5, contribution=0.8)
    local_rec = LocalExplanationRecord(
        sample_index=10,
        category="correct_positive",
        prediction=1,
        actual_value=1,
        base_value=0.5,
        contributions=[fc]
    )
    assert local_rec.sample_index == 10
    assert local_rec.category == "correct_positive"
    assert local_rec.prediction == 1
    assert len(local_rec.contributions) == 1


def test_explanation_output_serialization():
    out = ExplanationOutput(
        dataset_id="01_numerical_classification.csv",
        pipeline_id="classification_logistic_regression",
        model_name="LogisticRegression",
        task_type="classification",
        metric="f1",
        model_score=0.92,
        method="linear_coefficients",
        global_importance=[FeatureImportanceRecord(feature="f1", importance=1.0, rank=1)],
        local_explanations=[],
        warnings=["Test warning"]
    )
    dumped = out.model_dump()
    assert dumped["dataset_id"] == "01_numerical_classification.csv"
    assert dumped["method"] == "linear_coefficients"
    assert dumped["model_score"] == 0.92
    assert len(dumped["global_importance"]) == 1
