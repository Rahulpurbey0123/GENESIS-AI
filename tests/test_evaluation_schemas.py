"""
Unit Tests for GENESIS-AI Week 7 Evaluation Schemas.
"""

import pytest
from backend.evaluation.schemas import (
    BenchmarkConfig,
    RawObservation,
    AggregatedMetric,
    AblationRecord,
    StatisticalTestResult,
    HypothesisEvaluation,
    BenchmarkRunSummary,
    DatasetSpec
)


def test_dataset_spec_schema():
    spec = DatasetSpec(filename="01_numerical_classification.csv", target="target", task_type="classification")
    assert spec.filename == "01_numerical_classification.csv"
    assert spec.target == "target"
    assert spec.task_type == "classification"


def test_benchmark_config_defaults():
    config = BenchmarkConfig()
    assert config.max_evaluations == 200
    assert len(config.seeds) == 5
    assert "method_a_full_genesis" in config.methods


def test_raw_observation_schema():
    obs = RawObservation(
        dataset="01_numerical_classification.csv",
        task_type="classification",
        method="method_a_full_genesis",
        seed=42,
        metric="f1",
        score=0.95,
        candidate_evaluations=45,
        runtime_seconds=2.5,
        best_configuration={"pipeline_id": "classification_logistic_regression"}
    )
    assert obs.score == 0.95
    assert obs.status == "success"
    assert obs.error is None


def test_aggregated_metric_schema():
    agg = AggregatedMetric(
        dataset="01_numerical_classification.csv",
        method="method_a_full_genesis",
        task_type="classification",
        metric="f1",
        mean_score=0.95,
        std_score=0.01,
        best_score=0.96,
        worst_score=0.94,
        mean_evaluations=45.0,
        mean_runtime=2.5,
        success_count=5,
        fail_count=0
    )
    assert agg.mean_score == 0.95
    assert agg.fail_count == 0


def test_hypothesis_evaluation_schema():
    he = HypothesisEvaluation(
        hypothesis_id="H1",
        statement="GENESIS-AI achieves competitive predictive performance.",
        status="SUPPORTED",
        rationale="Empirical score equaled baseline."
    )
    assert he.status == "SUPPORTED"
