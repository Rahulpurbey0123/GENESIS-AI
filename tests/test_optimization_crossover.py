"""
Tests for Crossover Operator (backend/optimization/crossover.py).
"""

import pytest
import random
from backend.optimization.chromosome import Chromosome
from backend.optimization.crossover import crossover


def test_crossover_same_model_family():
    """Verify hyperparameter crossover between parents of the same model family."""
    p1 = Chromosome(
        pipeline_id="classification_random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}
    )
    p2 = Chromosome(
        pipeline_id="classification_random_forest",
        hyperparameters={"n_estimators": 500, "max_depth": 30, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "log2"}
    )

    rng = random.Random(42)
    off1, off2 = crossover(p1, p2, crossover_rate=1.0, rng=rng)

    assert off1.is_valid() is True
    assert off2.is_valid() is True
    assert off1.pipeline_id == "classification_random_forest"
    assert off2.pipeline_id == "classification_random_forest"


def test_crossover_different_model_family():
    """Verify structural crossover between parents of different model families."""
    p1 = Chromosome(pipeline_id="classification_logistic_regression", hyperparameters={"C": 0.01, "solver": "lbfgs", "max_iter": 500})
    p2 = Chromosome(pipeline_id="classification_svc", hyperparameters={"C": 10.0, "kernel": "rbf", "gamma": "scale"})

    rng = random.Random(42)
    off1, off2 = crossover(p1, p2, crossover_rate=1.0, rng=rng)

    assert off1.is_valid() is True
    assert off2.is_valid() is True
    assert off1.pipeline_id == "classification_svc"
    assert off2.pipeline_id == "classification_logistic_regression"
