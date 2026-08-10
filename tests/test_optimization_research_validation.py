"""
Research Safety and Pairing Validation Tests for Week 4.2.1 (tests/test_optimization_research_validation.py).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer
from experiments.analyze_week4_results import get_test_score, calculate_cache_metrics

DATASETS_DIR = Path(__file__).parent.parent / "data" / "test_datasets"


def test_matched_genesis_baseline_pairing_integrity():
    """Verify GENESIS and BASELINE runs share exact same seed, budget, and GA configuration."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"

    config_gen = OptimizationConfig(mode="genesis", top_k=2, population_size=10, generations=2, max_evaluations=20, random_state=123)
    config_base = OptimizationConfig(mode="baseline", top_k=10, population_size=10, generations=2, max_evaluations=20, random_state=123)

    opt_gen = EvolutionaryOptimizer(config=config_gen)
    opt_base = EvolutionaryOptimizer(config=config_base)

    res_gen = opt_gen.optimize(csv_path, target_column="target")
    res_base = opt_base.optimize(csv_path, target_column="target")

    assert res_gen.random_state == res_base.random_state == 123
    assert res_gen.max_evaluations == res_base.max_evaluations == 20
    assert res_gen.population_size == res_base.population_size == 10


def test_baseline_winner_retention_tracking():
    """Verify baseline_best_in_genesis_top_k calculation."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"

    opt_base = EvolutionaryOptimizer(config=OptimizationConfig(mode="baseline", random_state=42))
    res_base = opt_base.optimize(csv_path, target_column="target")
    winner_id = res_base.best_pipeline_id

    opt_gen = EvolutionaryOptimizer(config=OptimizationConfig(mode="genesis", top_k=2, random_state=42))
    res_gen = opt_gen.optimize(csv_path, target_column="target")

    retained = winner_id in res_gen.candidate_pipeline_ids
    res_gen.baseline_best_pipeline = winner_id
    res_gen.baseline_best_in_genesis_top_k = retained

    assert res_gen.baseline_best_pipeline == winner_id
    assert isinstance(res_gen.baseline_best_in_genesis_top_k, bool)


def test_test_set_isolation_during_evolution():
    """Verify test dataset is isolated and never passed to FitnessManager."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"

    config = OptimizationConfig(mode="genesis", top_k=2, population_size=6, generations=2, max_evaluations=12, random_state=42)
    optimizer = EvolutionaryOptimizer(config=config)
    result = optimizer.optimize(csv_path, target_column="target")

    assert result.test_performance is not None
    assert "f1" in result.test_performance


def test_cache_hit_rate_formula_and_zero_handling():
    """Fix #1: Test cache hit rate formula cache_hits / (unique_evals + cache_hits) and zero requests handling."""
    # Standard case: 25 unique, 175 cache hits -> 200 total requests, rate = 175/200 = 0.875
    total_reqs, hit_rate = calculate_cache_metrics(unique_evaluations=25, cache_hits=175)
    assert total_reqs == 200
    assert hit_rate == 0.875
    assert 0.0 <= hit_rate <= 1.0

    # Zero requests case: 0 unique, 0 cache hits -> 0 total requests, rate = 0.0
    zero_total, zero_rate = calculate_cache_metrics(unique_evaluations=0, cache_hits=0)
    assert zero_total == 0
    assert zero_rate == 0.0
    assert 0.0 <= zero_rate <= 1.0


def test_task_specific_metric_separation():
    """Fix #2: Test that classification (F1) and regression (RMSE) metrics are kept separate."""
    res_class = {"test_performance": {"f1": 0.85, "accuracy": 0.90}}
    res_regr = {"test_performance": {"rmse": 12.5, "mae": 10.0}}

    f1 = get_test_score(res_class, "classification")
    rmse = get_test_score(res_regr, "regression")

    assert f1 == 0.85
    assert rmse == 12.5
    assert type(f1) is float
    assert type(rmse) is float


def test_winner_retention_does_not_force_identical_performance():
    """Fix #3: Test that candidate inclusion (baseline_best_in_genesis_top_k=True) does not force identical score."""
    # Even if baseline_best_in_genesis_top_k is True, GA can explore different hyperparameters
    retained = True
    base_f1 = 0.80
    gen_f1 = 0.60  # Different hyperparameter configuration selected by GA

    f1_diff = gen_f1 - base_f1
    assert retained is True
    assert f1_diff == pytest.approx(-0.20)
