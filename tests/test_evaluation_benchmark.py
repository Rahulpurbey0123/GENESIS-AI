"""
Integration Unit Tests for GENESIS-AI Week 7 End-to-End Evaluation Pipeline.
"""

import pytest
from backend.evaluation.schemas import BenchmarkConfig, DatasetSpec
from backend.evaluation.runner import BenchmarkRunner
from backend.evaluation.statistics import StatisticsAnalyzer


def test_end_to_end_evaluation_pipeline(tmp_path):
    """Verify that runner, raw observation logging, aggregation, and hypothesis evaluation execute end-to-end."""
    bench_json = tmp_path / "week7_benchmark_results.json"

    spec1 = DatasetSpec(filename="01_numerical_classification.csv", target="target", task_type="classification")
    spec2 = DatasetSpec(filename="05_regression.csv", target="price", task_type="regression")

    config = BenchmarkConfig(
        datasets=[spec1, spec2],
        methods=["method_a_full_genesis", "method_d_recommendation_only"],
        seeds=[42],
        max_evaluations=10,
        population_size=5,
        generations=2,
        output_benchmark_json=str(bench_json)
    )

    runner = BenchmarkRunner(config=config)
    observations = runner.run_benchmark(verbose=False)

    assert len(observations) == 4
    for obs in observations:
        assert obs.status == "success"

    analyzer = StatisticsAnalyzer(observations, config=config)
    agg = analyzer.compute_aggregated_metrics()
    assert len(agg) == 4

    ablation = analyzer.compute_ablation_summary()
    assert len(ablation) == 2

    hypotheses = analyzer.evaluate_hypotheses()
    assert len(hypotheses) == 4
