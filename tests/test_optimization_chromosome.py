"""
Tests for Chromosome representation (backend/optimization/chromosome.py).
"""

import pytest
from backend.optimization.chromosome import Chromosome


def test_chromosome_creation_and_hash():
    """Verify chromosome creation, canonical hashing, and equality."""
    c1 = Chromosome(
        pipeline_id="classification_random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": 5}
    )
    c2 = Chromosome(
        pipeline_id="classification_random_forest",
        hyperparameters={"max_depth": 5, "n_estimators": 100}
    )

    assert c1.pipeline_id == "classification_random_forest"
    assert c1.get_canonical_key() == c2.get_canonical_key()
    assert c1.get_hash() == c2.get_hash()
    assert c1 == c2
    assert c1.is_valid() is True


def test_chromosome_serialization():
    """Verify to_dict and from_dict serialization."""
    c1 = Chromosome(
        pipeline_id="classification_logistic_regression",
        hyperparameters={"C": 1.0, "solver": "lbfgs", "max_iter": 500}
    )
    d = c1.to_dict()
    c2 = Chromosome.from_dict(d)

    assert c1 == c2
    assert c2.pipeline_id == "classification_logistic_regression"
    assert c2.hyperparameters["C"] == 1.0
