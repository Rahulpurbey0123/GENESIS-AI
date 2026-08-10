"""
End-to-end integration tests for backend/explainability/engine.py.
"""

import pytest
import numpy as np
import pandas as pd
from backend.optimization.evaluator import build_sklearn_pipeline
from backend.explainability.engine import ExplainabilityEngine


def test_engine_logistic_regression_pipeline():
    X_val = pd.DataFrame({"age": [25, 45, 35, 50, 22, 60], "fare": [10.5, 80.0, 30.0, 100.0, 7.5, 50.0]})
    y_val = pd.Series([0, 1, 0, 1, 0, 1])

    pipeline = build_sklearn_pipeline(
        pipeline_id="classification_logistic_regression",
        hyperparameters={"C": 1.0, "max_iter": 100},
        X_sample=X_val,
        random_state=42
    )
    pipeline.fit(X_val, y_val)

    engine = ExplainabilityEngine()
    out = engine.explain(
        pipeline_or_model=pipeline,
        X_val=X_val,
        y_val=y_val,
        dataset_id="test_logistic.csv",
        pipeline_id="classification_logistic_regression",
        task_type="classification",
        metric="f1",
        model_score=0.85
    )

    assert out.method == "linear_coefficients"
    assert len(out.global_importance) > 0
    assert len(out.local_explanations) > 0
    assert out.model_score == 0.85


def test_engine_random_forest_pipeline():
    X_val = pd.DataFrame({"x1": [1, 2, 3, 4, 5, 6], "x2": [6, 5, 4, 3, 2, 1]})
    y_val = pd.Series([0, 0, 0, 1, 1, 1])

    pipeline = build_sklearn_pipeline(
        pipeline_id="classification_random_forest",
        hyperparameters={"n_estimators": 10, "max_depth": 3},
        X_sample=X_val,
        random_state=42
    )
    pipeline.fit(X_val, y_val)

    engine = ExplainabilityEngine()
    out = engine.explain(
        pipeline_or_model=pipeline,
        X_val=X_val,
        y_val=y_val,
        dataset_id="test_rf.csv",
        pipeline_id="classification_random_forest",
        task_type="classification",
        metric="f1",
        model_score=0.90
    )

    assert out.method in ("shap_tree", "native_tree")
    assert len(out.global_importance) == 2
    assert len(out.local_explanations) > 0


def test_engine_svc_pipeline_permutation():
    X_val = pd.DataFrame({"v1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "v2": [2.0, 4.0, 1.0, 3.0, 5.0, 2.0]})
    y_val = pd.Series([0, 0, 0, 1, 1, 1])

    pipeline = build_sklearn_pipeline(
        pipeline_id="classification_svc",
        hyperparameters={"C": 1.0, "kernel": "rbf"},
        X_sample=X_val,
        random_state=42
    )
    pipeline.fit(X_val, y_val)

    engine = ExplainabilityEngine()
    out = engine.explain(
        pipeline_or_model=pipeline,
        X_val=X_val,
        y_val=y_val,
        dataset_id="test_svc.csv",
        pipeline_id="classification_svc",
        task_type="classification",
        metric="f1",
        model_score=0.88,
        n_repeats=3
    )

    assert out.method == "permutation_importance"
    assert len(out.global_importance) == 2
    assert out.local_explanations == []


def test_engine_regression_pipeline():
    X_val = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y_val = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])

    pipeline = build_sklearn_pipeline(
        pipeline_id="regression_linear_regression",
        hyperparameters={},
        X_sample=X_val,
        random_state=42
    )
    pipeline.fit(X_val, y_val)

    engine = ExplainabilityEngine()
    out = engine.explain(
        pipeline_or_model=pipeline,
        X_val=X_val,
        y_val=y_val,
        dataset_id="test_regression.csv",
        pipeline_id="regression_linear_regression",
        task_type="regression",
        metric="rmse",
        model_score=0.01
    )

    assert out.method == "linear_coefficients"
    assert len(out.global_importance) == 2
    assert len(out.local_explanations) > 0
