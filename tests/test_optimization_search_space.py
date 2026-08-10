"""
Tests for Discrete Search Spaces (backend/optimization/search_space.py).
"""

import pytest
import random
from backend.optimization.search_space import (
    SEARCH_SPACES,
    get_search_space,
    sample_random_hyperparameters,
    validate_hyperparameters
)


def test_get_search_space_valid_and_invalid():
    """Verify search space retrieval for valid and invalid pipeline_ids."""
    grid = get_search_space("classification_random_forest")
    assert "n_estimators" in grid
    assert 100 in grid["n_estimators"]

    with pytest.raises(ValueError, match="Unknown pipeline_id"):
        get_search_space("unknown_pipeline_id")


def test_sample_and_validate_hyperparameters():
    """Verify sampling random hyperparameters and validation logic."""
    rng = random.Random(42)
    pipeline_id = "classification_svc"

    hp = sample_random_hyperparameters(pipeline_id, rng=rng)
    assert "C" in hp
    assert "kernel" in hp
    assert "gamma" in hp
    assert validate_hyperparameters(pipeline_id, hp) is True

    # Invalid hyperparameter value
    invalid_hp = dict(hp)
    invalid_hp["C"] = 99999.0
    assert validate_hyperparameters(pipeline_id, invalid_hp) is False

    # Invalid hyperparameter name
    invalid_hp2 = dict(hp)
    invalid_hp2["non_existent_param"] = 10
    assert validate_hyperparameters(pipeline_id, invalid_hp2) is False
