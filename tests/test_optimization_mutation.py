"""
Tests for Mutation Operator (backend/optimization/mutation.py).
"""

import pytest
import random
from backend.optimization.chromosome import Chromosome
from backend.optimization.mutation import mutate
from backend.optimization.search_space import validate_hyperparameters


def test_mutation_produces_valid_chromosome():
    """Verify hyperparameter mutation produces valid chromosomes."""
    c = Chromosome(
        pipeline_id="classification_random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}
    )

    rng = random.Random(42)
    mutated = mutate(c, mutation_rate=1.0, pipeline_mutation_rate=0.0, rng=rng)

    assert mutated.is_valid() is True
    assert mutated.pipeline_id == "classification_random_forest"


def test_pipeline_mutation_changes_pipeline_id_and_regenerates_params():
    """Fix #2: Verify model-family pipeline mutation mutates pipeline_id and regenerates valid hyperparameters."""
    c = Chromosome(
        pipeline_id="classification_random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}
    )

    allowed = ["classification_random_forest", "classification_svc"]
    rng = random.Random(42)

    # Force pipeline mutation
    mutated = mutate(
        c,
        mutation_rate=0.0,
        pipeline_mutation_rate=1.0,
        allowed_pipeline_ids=allowed,
        rng=rng
    )

    assert mutated.pipeline_id == "classification_svc"
    assert "n_estimators" not in mutated.hyperparameters  # RF params purged
    assert "C" in mutated.hyperparameters                # SVC params regenerated
    assert "kernel" in mutated.hyperparameters
    assert validate_hyperparameters("classification_svc", mutated.hyperparameters) is True


def test_mutation_respects_allowed_candidate_pool():
    """Fix #2: Verify pipeline mutation NEVER selects a model family outside the mode's allowed candidate pool."""
    c = Chromosome(
        pipeline_id="classification_random_forest",
        hyperparameters={"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"}
    )

    # Pool restricted to RF and SVC
    allowed = ["classification_random_forest", "classification_svc"]
    rng = random.Random(42)

    for _ in range(50):
        mutated = mutate(
            c,
            mutation_rate=0.5,
            pipeline_mutation_rate=0.5,
            allowed_pipeline_ids=allowed,
            rng=rng
        )
        assert mutated.pipeline_id in allowed
        assert mutated.pipeline_id != "classification_logistic_regression"
        assert mutated.pipeline_id != "classification_k_neighbors"
