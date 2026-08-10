"""
Tests for Population Initialization (backend/optimization/population.py).
"""

import pytest
import random
from backend.recommendation.registry import PipelineRegistry
from backend.optimization.population import generate_initial_population


def test_generate_initial_population_valid():
    """Verify generation of initial population from candidate pipelines."""
    registry = PipelineRegistry()
    candidates = registry.get_pipelines_by_task("classification")
    rng = random.Random(42)

    pop = generate_initial_population(candidates, pop_size=20, rng=rng)

    assert len(pop) == 20
    for chrom in pop:
        assert chrom.is_valid() is True
        assert chrom.pipeline_id.startswith("classification_")


def test_generate_initial_population_empty_error():
    """Verify error raised when candidate pipelines list is empty."""
    with pytest.raises(ValueError, match="empty candidate pipelines"):
        generate_initial_population([], pop_size=10)
