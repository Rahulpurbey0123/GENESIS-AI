"""
Unit Tests for GENESIS-AI Week 7 Fairness Protocols.
"""

import pytest
from backend.evaluation.schemas import BenchmarkConfig, RawObservation
from backend.evaluation.protocols import FairnessProtocol


def test_fairness_protocol_valid_config():
    config = BenchmarkConfig()
    protocol = FairnessProtocol(config)
    is_valid, errors = protocol.validate_config()
    assert is_valid is True
    assert len(errors) == 0


def test_fairness_protocol_invalid_config():
    config = BenchmarkConfig(max_evaluations=-5)
    protocol = FairnessProtocol(config)
    is_valid, errors = protocol.validate_config()
    assert is_valid is False
    assert len(errors) > 0


def test_fairness_protocol_matched_runs():
    config = BenchmarkConfig()
    protocol = FairnessProtocol(config)

    obs1 = RawObservation(
        dataset="01_numerical_classification.csv",
        task_type="classification",
        method="method_a_full_genesis",
        seed=42,
        metric="f1",
        score=0.90,
        candidate_evaluations=30,
        runtime_seconds=1.2
    )

    obs2 = RawObservation(
        dataset="01_numerical_classification.csv",
        task_type="classification",
        method="method_e_unguided_baseline",
        seed=42,
        metric="f1",
        score=0.85,
        candidate_evaluations=200,
        runtime_seconds=5.4
    )

    is_matched, errs = protocol.validate_matched_runs(obs1, obs2)
    assert is_matched is True
    assert len(errs) == 0
