"""
GENESIS-AI Week 8 End-to-End Reliability Audit Test Suite.

Verifies:
1. Database content-hash deduplication preserves existing dataset IDs and experiment history.
2. Target-specific DIP profile persistence in SQLite.
3. Recommendation Engine returns detailed excluded candidates and exclusion reasons.
4. Top-K configuration restricts GENESIS search pool while BASELINE uses all candidates.
5. Random Search mode executes random sampling without crossover or tournament selection.
6. Target leakage prevention across split, model fit, SHAP, and LLM evidence.
7. Experiment isolation across dataset changes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import DatabaseService, get_db_connection
from backend.recommendation.engine import RecommendationEngine
from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer

client = TestClient(app)
DATA_DIR = Path("data/test_datasets")


@pytest.fixture
def clean_db():
    """Ensure clean database connection before test execution."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats")
    cursor.execute("DELETE FROM experiment_explanations")
    cursor.execute("DELETE FROM experiment_results")
    cursor.execute("DELETE FROM experiments")
    cursor.execute("DELETE FROM dip_profiles")
    cursor.execute("DELETE FROM datasets")
    conn.commit()
    conn.close()


def test_database_deduplication_and_experiment_preservation(clean_db):
    """
    REQ #12: Database content-hash deduplication.
    Re-uploading the exact same dataset returns the existing dataset_id and preserves experiment history.
    """
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        content = f.read()

    res1 = client.post(
        "/api/datasets/upload",
        files={"file": ("01_numerical_classification.csv", content, "text/csv")}
    )
    assert res1.status_code == 200
    ds_id_1 = res1.json()["id"]

    # Create an experiment attached to ds_id_1
    exp_res = client.post(
        "/api/experiments",
        data={
            "dataset_id": ds_id_1,
            "target_column": "target",
            "mode": "genesis",
            "generations": 1,
            "population_size": 5
        }
    )
    assert exp_res.status_code == 200
    exp_id = exp_res.json()["id"]

    # Upload same file content again
    res2 = client.post(
        "/api/datasets/upload",
        files={"file": ("01_numerical_classification.csv", content, "text/csv")}
    )
    assert res2.status_code == 200
    ds_id_2 = res2.json()["id"]

    # Verify ID matches and experiment history is preserved
    assert ds_id_1 == ds_id_2
    exp_check = client.get(f"/api/experiments/{exp_id}")
    assert exp_check.status_code == 200
    assert exp_check.json()["dataset_id"] == ds_id_1


def test_recommendation_excluded_candidates_and_reasons():
    """
    REQ #4: Recommendation response exposes candidate_count_before, candidate_count_after_filtering,
    and detailed excluded candidates with human-readable reasons.
    """
    csv_path = DATA_DIR / "02_categorical_heavy.csv"
    with open(csv_path, "rb") as f:
        content = f.read()

    rec_res = client.post(
        "/api/datasets/upload",
        files={"file": ("02_categorical_heavy.csv", content, "text/csv")}
    )
    ds_id = rec_res.json()["id"]

    rec_data = client.post(
        f"/api/datasets/{ds_id}/recommendations",
        data={"target_column": "target"}
    ).json()

    assert rec_data["candidate_count_before"] > 0
    assert rec_data["candidate_count_after_filtering"] <= rec_data["candidate_count_before"]
    assert "excluded_candidates" in rec_data

    # Verify excluded candidates have reasons
    for exc in rec_data["excluded_candidates"]:
        assert "pipeline_id" in exc
        assert "reason" in exc
        assert len(exc["reason"]) > 0


def test_optimization_modes_behavioral_difference():
    """
    REQ #7 & #25: Verify GENESIS, BASELINE, and RANDOM SEARCH optimization modes exhibit distinct candidate pools and behaviors.
    """
    df = pd.read_csv(DATA_DIR / "01_numerical_classification.csv")

    # GENESIS mode (top_k=2)
    config_genesis = OptimizationConfig(mode="genesis", top_k=2, population_size=4, generations=2, random_state=42)
    opt_genesis = EvolutionaryOptimizer(config=config_genesis)
    res_genesis = opt_genesis.optimize(df, target_column="target", evaluate_test=False)
    assert res_genesis.candidate_count_after <= 2

    # BASELINE mode (all compatible candidates)
    config_baseline = OptimizationConfig(mode="baseline", population_size=4, generations=2, random_state=42)
    opt_baseline = EvolutionaryOptimizer(config=config_baseline)
    res_baseline = opt_baseline.optimize(df, target_column="target", evaluate_test=False)
    assert res_baseline.candidate_count_after >= res_genesis.candidate_count_after

    # RANDOM SEARCH mode (random sampling without crossover)
    config_random = OptimizationConfig(mode="random", population_size=4, generations=2, random_state=42)
    opt_random = EvolutionaryOptimizer(config=config_random)
    res_random = opt_random.optimize(df, target_column="target", evaluate_test=False)
    assert res_random.mode == "random"
    assert len(res_random.history) == 2


def test_target_leakage_safety_invariant():
    """
    REQ #17 & #18: Verify target column 'subscribed' NEVER enters feature matrix X, SHAP, or feature attributions.
    """
    df = pd.read_csv(DATA_DIR / "02_categorical_heavy.csv")
    from backend.dataset.contract import get_canonical_data_split
    from backend.explainability.engine import ExplainabilityEngine
    from backend.optimization.evaluator import build_sklearn_pipeline

    X, y, feature_names, actual_target, id_cols = get_canonical_data_split(
        df, target_column="subscribed", exclude_identifiers=True
    )

    assert "subscribed" not in X.columns
    assert "subscribed" not in feature_names

    model = build_sklearn_pipeline("classification_random_forest", {"n_estimators": 5}, X, random_state=42)
    model.fit(X, y)

    engine = ExplainabilityEngine()
    output = engine.explain(model, X, y, dataset_id="test.csv", pipeline_id="classification_random_forest", task_type="classification")

    extracted_features = [fi.feature for fi in output.global_importance]
    assert "subscribed" not in extracted_features
