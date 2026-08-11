"""
Unit Tests for GENESIS-AI Week 7 Benchmark Runner.
"""

import pytest
from pathlib import Path
from backend.evaluation.schemas import BenchmarkConfig, DatasetSpec
from backend.evaluation.runner import BenchmarkRunner


def test_benchmark_runner_single_fast_run(tmp_path):
    """Test BenchmarkRunner on a fast mini-benchmark configuration."""
    test_json = tmp_path / "test_benchmark.json"

    spec = DatasetSpec(
        filename="01_numerical_classification.csv",
        target="target",
        task_type="classification"
    )

    config = BenchmarkConfig(
        datasets=[spec],
        methods=["method_d_recommendation_only"],
        seeds=[42],
        max_evaluations=5,
        population_size=5,
        generations=2,
        output_benchmark_json=str(test_json)
    )

    runner = BenchmarkRunner(config=config)
    observations = runner.run_benchmark(verbose=False)

    assert len(observations) == 1
    assert observations[0].status == "success"
    assert observations[0].method == "method_d_recommendation_only"
    assert test_json.exists()
