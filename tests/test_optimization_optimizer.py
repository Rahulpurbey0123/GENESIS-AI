"""
Integration and End-to-End Tests for EvolutionaryOptimizer (backend/optimization/optimizer.py).
"""

import pytest
from pathlib import Path
from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer, EvolutionaryOptimizerError


DATASETS_DIR = Path(__file__).parent.parent / "data" / "test_datasets"


def test_optimizer_end_to_end_classification():
    """Verify end-to-end optimization execution on 01_numerical_classification.csv."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"

    config = OptimizationConfig(
        mode="genesis",
        top_k=3,
        population_size=10,
        generations=3,
        max_evaluations=30,
        random_state=42
    )

    optimizer = EvolutionaryOptimizer(config=config)
    result = optimizer.optimize(csv_path, target_column="target")

    assert result.mode == "genesis"
    assert result.task_type == "classification"
    assert result.random_state == 42
    assert result.best_fitness >= 0.0
    assert result.best_pipeline_id is not None
    assert result.evaluations_used <= 30
    assert result.test_performance is not None
    assert "f1" in result.test_performance
    assert len(result.history) > 0


def test_optimizer_end_to_end_regression():
    """Verify end-to-end optimization execution on 05_regression.csv."""
    csv_path = DATASETS_DIR / "05_regression.csv"

    config = OptimizationConfig(
        mode="genesis",
        top_k=3,
        population_size=10,
        generations=3,
        max_evaluations=30,
        random_state=42
    )

    optimizer = EvolutionaryOptimizer(config=config)
    result = optimizer.optimize(csv_path, target_column="price")

    assert result.mode == "genesis"
    assert result.task_type == "regression"
    assert result.best_fitness <= 0.0  # Negated RMSE
    assert result.best_pipeline_id is not None
    assert result.test_performance is not None
    assert "rmse" in result.test_performance


def test_optimizer_reproducibility():
    """Verify that running optimization with same seed produces identical results."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"

    config = OptimizationConfig(
        mode="genesis",
        top_k=3,
        population_size=8,
        generations=2,
        max_evaluations=20,
        random_state=42
    )

    opt1 = EvolutionaryOptimizer(config=config)
    res1 = opt1.optimize(csv_path, target_column="target")

    opt2 = EvolutionaryOptimizer(config=config)
    res2 = opt2.optimize(csv_path, target_column="target")

    assert res1.best_pipeline_id == res2.best_pipeline_id
    assert res1.best_hyperparameters == res2.best_hyperparameters
    assert res1.best_fitness == res2.best_fitness
