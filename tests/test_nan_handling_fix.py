"""
Regression test suite for NaN, Infinity, Missing Values, and FAILED Experiment State handling.

Verifies:
1. Target containing NaN is cleaned by removing missing target rows (no target fabrication).
2. Target containing +/- infinity is cleaned by removing invalid target rows.
3. Numerical feature containing NaN / infinity is imputed during pipeline fit.
4. Categorical feature containing missing values is imputed during pipeline fit.
5. Mixed missing value tokens ("NA", "null", "", inf) are handled safely.
6. Database list_experiments handles FAILED status and None best_score without crashing.
7. Optimizer completes successfully when feature/target missingness is recoverable.
8. Optimizer fails gracefully when target contains only invalid/missing values.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from backend.dataset.cleaner import clean_dataset_for_ml
from backend.dataset.validator import DatasetValidationError
from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer, EvolutionaryOptimizerError
from backend.database import DatabaseService


def test_target_contains_nan_rows_removed():
    """Test A: Target contains NaN. Missing target rows must be removed, optimizer must not crash."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, np.nan, 1, 0, 1, 0, np.nan, 1, 0]
    })
    cleaned = clean_dataset_for_ml(df, "target")
    assert len(cleaned) == 8
    assert cleaned["target"].isna().sum() == 0

    config = OptimizationConfig(population_size=4, generations=2, max_evaluations=10, random_state=42)
    optimizer = EvolutionaryOptimizer(config=config)
    result = optimizer.optimize(cleaned, target_column="target", dataset_name="test_nan_target.csv")
    assert result.best_fitness != float("-inf")
    assert result.best_pipeline_id is not None


def test_target_contains_infinity_rows_removed():
    """Test B: Target contains infinity (+inf/-inf). Invalid target rows must be removed."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, np.inf, 1, -np.inf, 1, 0, 1, 0, 1]
    })
    cleaned = clean_dataset_for_ml(df, "target")
    assert len(cleaned) == 8
    assert cleaned["target"].isna().sum() == 0

    config = OptimizationConfig(population_size=4, generations=2, max_evaluations=10, random_state=42)
    optimizer = EvolutionaryOptimizer(config=config)
    result = optimizer.optimize(cleaned, target_column="target", dataset_name="test_inf_target.csv")
    assert result.best_fitness != float("-inf")


def test_numerical_feature_contains_nan_and_inf():
    """Test C: Numerical feature contains NaN and +/- infinity. Feature imputation occurs."""
    df = pd.DataFrame({
        "feature1": [1.0, np.nan, 3.0, np.inf, 5.0, -np.inf, 7.0, 8.0, 9.0, 10.0],
        "feature2": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    cleaned = clean_dataset_for_ml(df, "target")
    assert len(cleaned) == 10
    # Confirm inf values converted to NaN
    assert np.isinf(cleaned["feature1"]).sum() == 0

    config = OptimizationConfig(population_size=4, generations=2, max_evaluations=10, random_state=42)
    optimizer = EvolutionaryOptimizer(config=config)
    result = optimizer.optimize(cleaned, target_column="target", dataset_name="test_num_feature_nan.csv")
    assert result.best_fitness != float("-inf")


def test_categorical_feature_contains_missing_values():
    """Test D: Categorical feature contains missing values / empty strings."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "cat_feature": ["A", "B", None, "A", "", "B", "A", "NA", "B", "A"],
        "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    })
    cleaned = clean_dataset_for_ml(df, "target")
    assert len(cleaned) == 10

    config = OptimizationConfig(population_size=4, generations=2, max_evaluations=10, random_state=42)
    optimizer = EvolutionaryOptimizer(config=config)
    result = optimizer.optimize(cleaned, target_column="target", dataset_name="test_cat_feature_nan.csv")
    assert result.best_fitness != float("-inf")


def test_mixed_missing_values_and_tokens():
    """Test E: Mixed missing values (NaN, inf, -inf, empty string, NA, null)."""
    df = pd.DataFrame({
        "num1": [1.0, np.nan, np.inf, 4.0, -np.inf, 6.0, 7.0, 8.0, 9.0, 10.0],
        "cat1": ["cat", "dog", None, "", "NA", "N/A", "cat", "dog", "cat", "dog"],
        "target": ["classA", "classB", "classA", "classB", "NA", "classA", "classB", "classA", "classB", "classA"]
    })
    cleaned = clean_dataset_for_ml(df, "target")
    # Row index 4 had target = "NA", so row is removed -> 9 rows remain
    assert len(cleaned) == 9
    assert cleaned["target"].isna().sum() == 0

    config = OptimizationConfig(population_size=4, generations=2, max_evaluations=10, random_state=42)
    optimizer = EvolutionaryOptimizer(config=config)
    result = optimizer.optimize(cleaned, target_column="target", dataset_name="test_mixed_nan.csv")
    assert result.best_fitness != float("-inf")


import uuid


def test_failed_experiment_state_in_database():
    """Test F & G: Failed experiment state in database list_experiments without crash."""
    exp_id = f"test_fail_{uuid.uuid4().hex[:10]}"
    record = DatabaseService.create_experiment(

        experiment_id=exp_id,
        dataset_id="ds_test",
        dataset_name="test.csv",
        target_column="target",
        mode="genesis",
        config={"generations": 5}
    )
    assert record["status"] == "RUNNING"

    # Simulate job failure
    DatabaseService.update_experiment_progress(
        experiment_id=exp_id,
        status="FAILED",
        progress={
            "current_generation": 0,
            "max_generations": 5,
            "evaluated_pipelines": 0,
            "best_score": None,
            "runtime": 0.0
        },
        error_message="Target contains missing or invalid values."
    )

    history = DatabaseService.list_experiments()
    failed_item = next((item for item in history if item["id"] == exp_id), None)
    assert failed_item is not None
    assert failed_item["status"] == "FAILED"
    assert failed_item["best_score"] is None
    assert failed_item["error_message"] == "Target contains missing or invalid values."


def test_optimizer_fails_gracefully_when_target_all_nan():
    """Test H: When target column contains 100% missing values, clean_dataset_for_ml raises DatasetValidationError."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "target": [np.nan, np.nan, None, np.nan]
    })
    with pytest.raises(DatasetValidationError) as excinfo:
        clean_dataset_for_ml(df, "target")
    assert "Target column 'target' contains only missing/invalid values" in str(excinfo.value)


def test_single_final_test_evaluation_without_leakage():
    """Test Group B: Single final test evaluation & test set leakage prevention."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    config = OptimizationConfig(population_size=4, generations=2, max_evaluations=10, random_state=42)
    optimizer = EvolutionaryOptimizer(config=config)

    # When evaluate_test=False is passed (used by jobs.py), optimizer skips duplicate test evaluation
    res_no_test = optimizer.optimize(df, target_column="target", dataset_name="test_no_leak.csv", evaluate_test=False)
    assert res_no_test.test_performance == {}
    assert res_no_test.best_pipeline_id is not None
    assert res_no_test.best_fitness != float("-inf")


def test_failure_progress_preservation():
    """Test Group C: Failure progress preservation. Real progress is retained on failure."""
    exp_id = f"test_prog_fail_{uuid.uuid4().hex[:10]}"
    DatabaseService.create_experiment(
        experiment_id=exp_id,
        dataset_id="ds_test",
        dataset_name="test.csv",
        target_column="target",
        mode="genesis",
        config={"generations": 10}
    )

    # Simulate real progress updates during GA loop
    DatabaseService.update_experiment_progress(
        experiment_id=exp_id,
        status="RUNNING",
        progress={
            "current_generation": 3,
            "max_generations": 10,
            "evaluated_pipelines": 20,
            "best_score": 0.85,
            "runtime": 12.5,
            "search_space_reduction": 0.40,
            "history": [{"gen": 1, "best_score": 0.80}, {"gen": 2, "best_score": 0.82}, {"gen": 3, "best_score": 0.85}]
        }
    )

    # Simulate a mid-run job failure (e.g., exception during final fit/eval)
    from backend.jobs import ExperimentJob
    existing_exp = DatabaseService.get_experiment(exp_id)
    current_progress = existing_exp.get("progress", {})

    failed_progress = {
        "current_generation": current_progress.get("current_generation", 0),
        "max_generations": current_progress.get("max_generations", 10),
        "evaluated_pipelines": current_progress.get("evaluated_pipelines", 0),
        "best_score": current_progress.get("best_score", None),
        "runtime": current_progress.get("runtime", 0.0),
        "search_space_reduction": current_progress.get("search_space_reduction", 0.0),
        "history": current_progress.get("history", [])
    }

    DatabaseService.update_experiment_progress(
        experiment_id=exp_id,
        status="FAILED",
        progress=failed_progress,
        error_message="Runtime error during post-GA evaluation"
    )

    fetched = DatabaseService.get_experiment(exp_id)
    assert fetched["status"] == "FAILED"
    assert fetched["progress"]["current_generation"] == 3
    assert fetched["progress"]["evaluated_pipelines"] == 20
    assert fetched["progress"]["runtime"] == 12.5
    assert len(fetched["progress"]["history"]) == 3
    assert fetched["error_message"] == "Runtime error during post-GA evaluation"

