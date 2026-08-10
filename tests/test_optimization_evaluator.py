"""
Tests for Pipeline Evaluator (backend/optimization/evaluator.py).
"""

import pytest
import pandas as pd
import numpy as np
from backend.optimization.chromosome import Chromosome
from backend.optimization.evaluator import evaluate_chromosome, build_sklearn_pipeline


def test_evaluate_classification_chromosome():
    """Verify evaluation of a classification chromosome on synthetic data."""
    c = Chromosome(
        pipeline_id="classification_logistic_regression",
        hyperparameters={"C": 1.0, "solver": "lbfgs", "max_iter": 500}
    )

    X_train = pd.DataFrame({"x1": np.random.randn(50), "x2": np.random.randn(50)})
    y_train = pd.Series(np.random.choice([0, 1], size=50))
    X_val = pd.DataFrame({"x1": np.random.randn(20), "x2": np.random.randn(20)})
    y_val = pd.Series(np.random.choice([0, 1], size=20))

    fitness, metrics = evaluate_chromosome(
        chromosome=c,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type="classification"
    )

    assert 0.0 <= fitness <= 1.0
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_evaluate_regression_chromosome():
    """Verify evaluation of a regression chromosome on synthetic data."""
    c = Chromosome(
        pipeline_id="regression_linear_regression",
        hyperparameters={"fit_intercept": True}
    )

    X_train = pd.DataFrame({"x1": np.random.randn(50)})
    y_train = pd.Series(np.random.randn(50))
    X_val = pd.DataFrame({"x1": np.random.randn(20)})
    y_val = pd.Series(np.random.randn(20))

    fitness, metrics = evaluate_chromosome(
        chromosome=c,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type="regression"
    )

    # GA maximizes fitness, so regression fitness = -RMSE <= 0
    assert fitness <= 0.0
    assert "rmse" in metrics
    assert "mae" in metrics
