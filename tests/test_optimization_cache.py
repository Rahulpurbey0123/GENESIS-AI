"""
Tests for Evaluation Cache (backend/optimization/cache.py).
"""

import pytest
from backend.optimization.chromosome import Chromosome
from backend.optimization.cache import EvaluationCache


def test_evaluation_cache_hits_and_misses():
    """Verify cache put, get, hits, and misses metrics."""
    cache = EvaluationCache()
    c1 = Chromosome(
        pipeline_id="classification_logistic_regression",
        hyperparameters={"C": 1.0, "solver": "lbfgs", "max_iter": 500}
    )

    # Initial miss
    assert cache.contains(c1) is False
    assert cache.get(c1) is None
    assert cache.evaluation_requests == 1
    assert cache.cache_hits == 0

    # Put in cache
    cache.put(c1, fitness=0.85, metrics={"accuracy": 0.85})
    assert cache.contains(c1) is True
    assert cache.size() == 1

    # Cache hit
    cached_val = cache.get(c1)
    assert cached_val is not None
    assert cached_val["fitness"] == 0.85
    assert cache.evaluation_requests == 2
    assert cache.cache_hits == 1
