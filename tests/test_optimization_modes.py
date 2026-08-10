"""
Tests for GENESIS Mode vs BASELINE Mode (backend/optimization/modes.py).
"""

import pytest
from pathlib import Path
from backend.optimization.modes import run_genesis_mode, run_baseline_mode


DATASETS_DIR = Path(__file__).parent.parent / "data" / "test_datasets"


def test_genesis_reduces_candidate_space():
    """Fix #1: Verify GENESIS mode reduces candidate space (e.g. 5->2, 60%) while BASELINE is 0%."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"

    # GENESIS mode with top_k=2 on 5 compatible candidates
    res_gen = run_genesis_mode(
        file_source_or_df=csv_path,
        target_column="target",
        top_k=2,
        population_size=10,
        generations=2,
        max_evaluations=20,
        random_state=42
    )

    # BASELINE mode with all 5 compatible candidates
    res_base = run_baseline_mode(
        file_source_or_df=csv_path,
        target_column="target",
        population_size=10,
        generations=2,
        max_evaluations=20,
        random_state=42
    )

    assert res_gen.candidate_count_before == 5
    assert res_gen.candidate_count_after == 2
    assert res_gen.candidate_space_reduction == 0.60
    assert len(res_gen.candidate_pipeline_ids) == 2

    assert res_base.candidate_count_before == 5
    assert res_base.candidate_count_after == 5
    assert res_base.candidate_space_reduction == 0.0
    assert len(res_base.candidate_pipeline_ids) == 5


def test_genesis_and_baseline_use_same_evaluation_budget():
    """Verify GENESIS and BASELINE modes receive exact same max_evaluations budget."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"

    res_gen = run_genesis_mode(csv_path, target_column="target", top_k=2, max_evaluations=20, random_state=42)
    res_base = run_baseline_mode(csv_path, target_column="target", max_evaluations=20, random_state=42)

    assert res_gen.max_evaluations == 20
    assert res_base.max_evaluations == 20


def test_genesis_and_baseline_use_same_ga_configuration():
    """Verify GENESIS and BASELINE modes use identical GA configuration parameters."""
    csv_path = DATASETS_DIR / "01_numerical_classification.csv"

    res_gen = run_genesis_mode(csv_path, target_column="target", top_k=2, population_size=12, generations=3, random_state=42)
    res_base = run_baseline_mode(csv_path, target_column="target", population_size=12, generations=3, random_state=42)

    assert res_gen.population_size == res_base.population_size == 12
    assert res_gen.generations == res_base.generations == 3
    assert res_gen.random_state == res_base.random_state == 42
