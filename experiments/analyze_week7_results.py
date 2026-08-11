"""
GENESIS-AI Week 7 Results Analysis & Hypothesis Evaluation Script.

Loads raw benchmark observations from experiments/week7_benchmark_results.json,
validates schema, aggregates repeated seed runs, computes ablation breakdowns,
performs statistical paired tests (t-tests, effect sizes), evaluates hypotheses H1-H4,
exports experiments/week7_summary.csv and experiments/week7_ablation_results.json,
and prints structured findings.
"""

import sys
import json
import csv
import platform
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in Python path
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.evaluation.schemas import (
    RawObservation,
    AggregatedMetric,
    AblationRecord,
    BenchmarkRunSummary,
    BenchmarkConfig
)
from backend.evaluation.configuration import get_default_benchmark_config, METHOD_METADATA
from backend.evaluation.statistics import StatisticsAnalyzer


def export_summary_csv(aggregated_metrics: List[AggregatedMetric], output_path: str):
    """Export aggregated metrics table to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset", "method", "task_type", "metric", "mean_score", "std_score",
        "best_score", "worst_score", "mean_evaluations", "mean_runtime",
        "success_count", "fail_count"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in aggregated_metrics:
            writer.writerow(metric.model_dump())

    print(f"Exported summary CSV to: {path}")


def export_ablation_json(ablation_summary: List[AblationRecord], output_path: str):
    """Export ablation summary table to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [rec.model_dump() for rec in ablation_summary]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Exported ablation JSON to: {path}")


def main():
    print("=" * 100)
    print("GENESIS-AI WEEK 7 RESEARCH ANALYSIS & HYPOTHESIS EVALUATION")
    print("=" * 100)

    config = get_default_benchmark_config()
    raw_path = Path(config.output_benchmark_json)

    if not raw_path.exists():
        print(f"ERROR: Raw benchmark result file not found at: {raw_path}")
        print("Please run `python experiments/run_week7_benchmarks.py` first.")
        sys.exit(1)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    observations = [RawObservation(**item) for item in raw_json]
    print(f"Loaded {len(observations)} raw observations from {raw_path}")

    analyzer = StatisticsAnalyzer(observations, config=config)

    # 1. Aggregated Metrics per Dataset and Method
    aggregated_metrics = analyzer.compute_aggregated_metrics()
    export_summary_csv(aggregated_metrics, config.output_summary_csv)
    export_summary_csv(aggregated_metrics, str(Path(config.output_summary_csv).parent / "week7_summary_v1_2.csv"))
    export_summary_csv(aggregated_metrics, str(Path(config.output_summary_csv).parent / "week7_summary_v1_3.csv"))
    export_summary_csv(aggregated_metrics, str(Path(config.output_summary_csv).parent / "week7_summary_v1_4.csv"))

    # 2. Ablation Analysis Breakdown
    ablation_summary = analyzer.compute_ablation_summary()
    export_ablation_json(ablation_summary, config.output_ablation_json)
    export_ablation_json(ablation_summary, str(Path(config.output_ablation_json).parent / "week7_ablation_results_v1_2.json"))
    export_ablation_json(ablation_summary, str(Path(config.output_ablation_json).parent / "week7_ablation_results_v1_3.json"))
    export_ablation_json(ablation_summary, str(Path(config.output_ablation_json).parent / "week7_ablation_results_v1_4.json"))

    # Also save v1_2, v1_3, and v1_4 raw benchmark copy
    v1_2_raw_path = raw_path.parent / "week7_benchmark_results_v1_2.json"
    v1_3_raw_path = raw_path.parent / "week7_benchmark_results_v1_3.json"
    v1_4_raw_path = raw_path.parent / "week7_benchmark_results_v1_4.json"
    with open(v1_2_raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_json, f, indent=2)
    with open(v1_3_raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_json, f, indent=2)
    with open(v1_4_raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_json, f, indent=2)
    print(f"Exported v1_4 benchmark raw JSON to: {v1_4_raw_path}")

    # 3. Statistical Testing
    stat_tests = analyzer.perform_statistical_tests()

    # 4. Hypothesis Evaluations
    hypotheses = analyzer.evaluate_hypotheses()

    print("\n" + "=" * 110)
    print("ABLATION BREAKDOWN (SUMMARY ACROSS DATASETS & SEEDS)")
    print("=" * 110)
    print(f"{'Code':<5} | {'Method':<32} | {'Class F1':<10} | {'Reg RMSE':<10} | {'Mean Evals':<10} | {'Mean Time':<10} | {'Rel Efficiency'}")
    print("-" * 110)
    for rec in ablation_summary:
        print(f"{rec.method_code:<5} | {rec.method:<32} | {rec.mean_classification_f1:<10.4f} | {rec.mean_regression_rmse:<10.2f} | {rec.mean_candidate_evaluations:<10.1f} | {rec.mean_runtime_seconds:<10.2f}s | {rec.relative_efficiency_gain:+.1f}%")

    print("\n" + "=" * 100)
    print("HYPOTHESIS EVALUATION SUMMARY")
    print("=" * 100)
    for h in hypotheses:
        print(f"[{h.hypothesis_id}] {h.statement}")
        print(f"     STATUS: {h.status}")
        print(f"     RATIONALE: {h.rationale}")
        print("-" * 100)

    print("\n" + "=" * 100)
    print("STATISTICAL COMPARISONS")
    print("=" * 100)
    for test in stat_tests:
        print(f"Test: {test.test_name} ({test.comparison})")
        print(f"     {test.interpretation}")
        print("-" * 100)


if __name__ == "__main__":
    main()
