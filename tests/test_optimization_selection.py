"""
Tests for Tournament Selection Operator (backend/optimization/selection.py).
"""

import pytest
import random
from backend.optimization.chromosome import Chromosome
from backend.optimization.selection import tournament_selection


def test_tournament_selection_deterministic():
    """Verify tournament selection selects higher fitness parents deterministically."""
    c1 = Chromosome(pipeline_id="classification_logistic_regression", hyperparameters={"C": 0.01, "solver": "lbfgs", "max_iter": 500})
    c2 = Chromosome(pipeline_id="classification_random_forest", hyperparameters={"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"})
    c3 = Chromosome(pipeline_id="classification_svc", hyperparameters={"C": 1.0, "kernel": "rbf", "gamma": "scale"})

    pop = [c1, c2, c3]
    fitnesses = [0.20, 0.95, 0.50]

    rng = random.Random(42)
    selected = tournament_selection(pop, fitnesses, tournament_size=3, num_select=10, rng=rng)

    assert len(selected) == 10
    # c2 has highest fitness (0.95), so in tournament size 3 c2 should dominate selection
    c2_count = sum(1 for s in selected if s == c2)
    assert c2_count > 5
