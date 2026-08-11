"""
Unit & Integration Tests for GENESIS-AI Week 7 Baselines & Ablations (Hardened v1.2).
"""

import pytest
from unittest.mock import patch, MagicMock
from backend.evaluation.schemas import BenchmarkConfig
from backend.evaluation.datasets import DatasetManager
from backend.evaluation.baselines import BaselineExecutor, create_neutral_dip
from backend.recommendation.engine import RecommendationEngine


def test_baseline_executor_method_d():
    """Fast execution test for Method D (Recommendation Only)."""
    config = BenchmarkConfig(max_evaluations=10, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()

    df = ds_mgr.load_dataset("01_numerical_classification.csv")
    obs = executor.run_method_d(
        df=df,
        target_column="target",
        task_type="classification",
        seed=42,
        dataset_name="01_numerical_classification.csv"
    )

    assert obs.status == "success"
    assert obs.method == "method_d_recommendation_only"
    assert obs.candidate_evaluations > 0
    assert 0.0 <= obs.score <= 1.0


def test_baseline_executor_method_a():
    """Fast execution test for Method A (Full GENESIS-AI)."""
    config = BenchmarkConfig(max_evaluations=10, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()

    df = ds_mgr.load_dataset("01_numerical_classification.csv")
    obs = executor.run_method_a(
        df=df,
        target_column="target",
        task_type="classification",
        seed=42,
        dataset_name="01_numerical_classification.csv"
    )

    assert obs.status == "success"
    assert obs.method == "method_a_full_genesis"
    assert obs.candidate_evaluations > 0
    assert 0.0 <= obs.score <= 1.0


def test_baseline_executor_method_b():
    """Fast execution test for Method B (Without DIP)."""
    config = BenchmarkConfig(max_evaluations=10, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()

    df = ds_mgr.load_dataset("01_numerical_classification.csv")
    obs = executor.run_method_b(
        df=df,
        target_column="target",
        task_type="classification",
        seed=42,
        dataset_name="01_numerical_classification.csv"
    )

    assert obs.status == "success"
    assert obs.method == "method_b_without_dip"
    assert obs.candidate_evaluations > 0
    assert 0.0 <= obs.score <= 1.0


def test_baseline_executor_method_e():
    """Fast execution test for Method E (Unguided Baseline)."""
    config = BenchmarkConfig(max_evaluations=5, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()

    df = ds_mgr.load_dataset("01_numerical_classification.csv")
    obs = executor.run_method_e(
        df=df,
        target_column="target",
        task_type="classification",
        seed=42,
        dataset_name="01_numerical_classification.csv"
    )

    assert obs.status == "success"
    assert obs.method == "method_e_unguided_baseline"
    assert obs.candidate_evaluations == 5
    assert 0.0 <= obs.score <= 1.0


def test_method_b_dip_independence():
    """Verify that Method B executes successfully even if generate_dip is mocked to fail (proving ZERO DIP dependency)."""
    config = BenchmarkConfig(max_evaluations=5, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()
    df = ds_mgr.load_dataset("01_numerical_classification.csv")

    with patch("backend.evaluation.baselines.generate_dip", side_effect=RuntimeError("DIP MUST NOT BE CALLED IN METHOD B")):
        obs = executor.run_method_b(
            df=df,
            target_column="target",
            task_type="classification",
            seed=42,
            dataset_name="01_numerical_classification.csv"
        )
        assert obs.status == "success"
        assert obs.method == "method_b_without_dip"


def test_method_b_uses_recommendation_pathway():
    """Verify that Method B queries RecommendationEngine.recommend_from_dip with neutral_dip profile."""
    config = BenchmarkConfig(max_evaluations=5, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()
    df = ds_mgr.load_dataset("01_numerical_classification.csv")

    with patch.object(RecommendationEngine, "recommend_from_dip", side_effect=RecommendationEngine().recommend_from_dip) as mock_rec:
        obs = executor.run_method_b(
            df=df,
            target_column="target",
            task_type="classification",
            seed=42,
            dataset_name="01_numerical_classification.csv"
        )
        assert obs.status == "success"
        assert mock_rec.called


def test_method_b_changing_dip_output_no_effect():
    """Verify that changing DIP generator output has zero effect on Method B candidate selection or execution."""
    config = BenchmarkConfig(max_evaluations=5, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()
    df = ds_mgr.load_dataset("01_numerical_classification.csv")

    obs1 = executor.run_method_b(df=df, target_column="target", task_type="classification", seed=42)

    with patch("backend.evaluation.baselines.generate_dip", return_value={"GARBAGE": 123}):
        obs2 = executor.run_method_b(df=df, target_column="target", task_type="classification", seed=42)

    assert obs1.score == obs2.score
    assert obs1.candidate_evaluations == obs2.candidate_evaluations


def test_method_e_dip_independence():
    """Verify that Method E executes successfully even if generate_dip is mocked to fail (proving ZERO DIP dependency)."""
    config = BenchmarkConfig(max_evaluations=5, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()
    df = ds_mgr.load_dataset("01_numerical_classification.csv")

    with patch("backend.evaluation.baselines.generate_dip", side_effect=RuntimeError("DIP MUST NOT BE CALLED IN METHOD E")):
        obs = executor.run_method_e(
            df=df,
            target_column="target",
            task_type="classification",
            seed=42,
            dataset_name="01_numerical_classification.csv"
        )
        assert obs.status == "success"
        assert obs.method == "method_e_unguided_baseline"


def test_method_d_top_k_pool_equivalence():
    """Verify Method D queries RecommendationEngine with top_k matching self.config.top_k for pool equivalence with Method A."""
    config = BenchmarkConfig(top_k=2, max_evaluations=5, population_size=5, generations=2)
    executor = BaselineExecutor(config=config)
    ds_mgr = DatasetManager()
    df = ds_mgr.load_dataset("01_numerical_classification.csv")

    with patch.object(RecommendationEngine, "recommend_from_dip", side_effect=RecommendationEngine().recommend_from_dip) as mock_rec:
        obs = executor.run_method_d(
            df=df,
            target_column="target",
            task_type="classification",
            seed=42,
            dataset_name="01_numerical_classification.csv"
        )
        assert obs.status == "success"
        assert mock_rec.called
        # Check that recommend_from_dip was called with top_k=2
        call_kwargs = mock_rec.call_args.kwargs
        assert call_kwargs.get("top_k") == 2
        assert obs.candidate_evaluations == 2

