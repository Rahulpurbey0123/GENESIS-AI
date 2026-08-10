"""
Tests for Fitness Manager and Budget Enforcer (backend/optimization/fitness.py).
"""

import pytest
import pandas as pd
import numpy as np
from backend.optimization.chromosome import Chromosome
from backend.optimization.cache import EvaluationCache
from backend.optimization.fitness import FitnessManager
from backend.optimization.evaluator import evaluate_chromosome


def test_fitness_manager_budget_enforcement():
    """Verify max_evaluations budget stopping and fallback float('-inf') behavior."""
    X_train = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y_train = pd.Series([0, 1, 0, 1, 0])
    X_val = pd.DataFrame({"x": [1, 2]})
    y_val = pd.Series([0, 1])

    cache = EvaluationCache()
    # Set max_evaluations = 2
    fm = FitnessManager(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type="classification",
        max_evaluations=2,
        cache=cache
    )

    c1 = Chromosome("classification_logistic_regression", {"C": 0.01, "solver": "lbfgs", "max_iter": 500})
    c2 = Chromosome("classification_logistic_regression", {"C": 0.1, "solver": "lbfgs", "max_iter": 500})
    c3 = Chromosome("classification_logistic_regression", {"C": 1.0, "solver": "lbfgs", "max_iter": 500})

    fitnesses, _ = fm.evaluate_population([c1, c2, c3])

    assert fm.evaluations_used == 2
    assert fitnesses[0] != float("-inf")
    assert fitnesses[1] != float("-inf")
    assert fitnesses[2] == float("-inf")  # Exceeded budget


def test_failed_regression_fitness_returns_neg_inf():
    """Fix #3: Verify failed regression evaluation returns float('-inf') and valid fitness beats float('-inf')."""
    # Trigger exception via invalid parameter choice for estimator
    invalid_chrom = Chromosome("regression_linear_regression", {"fit_intercept": "invalid_boolean_string"})

    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y_train = pd.Series([10.0, 20.0, 30.0])
    X_val = pd.DataFrame({"x": [1.5]})
    y_val = pd.Series([15.0])

    fitness, metrics = evaluate_chromosome(
        chromosome=invalid_chrom,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type="regression"
    )

    assert fitness == float("-inf")
    assert metrics["status"] == "failed"

    valid_fitness = -5000.0  # High RMSE
    assert valid_fitness > fitness


def test_failed_classification_fitness_returns_neg_inf():
    """Fix #3: Verify failed classification evaluation returns float('-inf') and valid F1 beats float('-inf')."""
    # Trigger exception via invalid solver choice for LogisticRegression
    invalid_chrom = Chromosome("classification_logistic_regression", {"solver": "invalid_solver", "C": 1.0, "max_iter": 500})

    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 1, 0])
    X_val = pd.DataFrame({"x": [1.5]})
    y_val = pd.Series([1])

    fitness, metrics = evaluate_chromosome(
        chromosome=invalid_chrom,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task_type="classification"
    )

    assert fitness == float("-inf")
    assert metrics["status"] == "failed"

    valid_fitness = 0.5
    assert valid_fitness > fitness
