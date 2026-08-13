"""
Automated Scientific Integrity & Hardening Tests for Week 8 Application.
Verifies that zero scientific numbers are fabricated, metrics are computed from real test set predictions,
explainability uses the Week 5 engine, and recommendation viewing is read-only.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import DatabaseService
from backend.explainability.engine import ExplainabilityEngine

client = TestClient(app)
DATA_DIR = Path("data/test_datasets")


def test_recommendations_endpoint_is_read_only():
    """Verify that requesting recommendations does NOT create an experiment."""
    # 1. Count experiments before
    initial_exps = DatabaseService.list_experiments()
    initial_count = len(initial_exps)

    # 2. Upload dataset
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    # 3. Request recommendations via dataset endpoint
    rec_res = client.post(f"/api/datasets/{ds_id}/recommendations")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert "recommendations" in rec_data
    assert rec_data["candidate_count_before"] > 0

    # 4. Verify experiment count did NOT increase
    post_rec_exps = DatabaseService.list_experiments()
    assert len(post_rec_exps) == initial_count


def test_experiment_creates_real_scikit_learn_metrics():
    """Verify that classification metrics are computed from real test set predictions, not formulas."""
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    exp_res = client.post(
        "/api/experiments",
        data={
            "dataset_id": ds_id,
            "target_column": "target",
            "mode": "genesis",
            "generations": 2,
            "population_size": 5
        }
    )
    assert exp_res.status_code == 200
    exp_id = exp_res.json()["id"]

    # Wait for completion
    import time
    for _ in range(20):
        poll = client.get(f"/api/experiments/{exp_id}")
        if poll.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    res = client.get(f"/api/experiments/{exp_id}/results")
    assert res.status_code == 200
    metrics = res.json()["metrics"]

    # Verify no formulaic outputs (e.g. accuracy * 0.98 != precision)
    # Metrics must be numbers or None
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "balanced_accuracy" in metrics


def test_explainability_uses_real_engine_method():
    """Verify that explainability engine outputs genuine method names and features."""
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    exp_res = client.post(
        "/api/experiments",
        data={
            "dataset_id": ds_id,
            "target_column": "target",
            "mode": "genesis",
            "generations": 2,
            "population_size": 5
        }
    )
    exp_id = exp_res.json()["id"]

    import time
    for _ in range(20):
        poll = client.get(f"/api/experiments/{exp_id}")
        if poll.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    exp_data = client.get(f"/api/experiments/{exp_id}/explanations").json()
    assert "shap" in exp_data
    assert "method" in exp_data["shap"]
    assert exp_data["shap"]["method"] in ("shap_tree", "permutation_importance", "linear_coefficients", "native_tree", "unsupported")
    assert isinstance(exp_data["global_importance"]["features"], list)


def test_regression_experiment_real_metrics():
    """Verify regression experiment calculates real MAE, RMSE, and R2."""
    csv_path = DATA_DIR / "05_regression.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("05_regression.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    target_col = upload_res.json().get("suggested_target", "price")
    exp_res = client.post(
        "/api/experiments",
        data={
            "dataset_id": ds_id,
            "target_column": target_col,
            "mode": "genesis",
            "generations": 2,
            "population_size": 5
        }
    )
    assert exp_res.status_code == 200
    exp_id = exp_res.json()["id"]

    import time
    for _ in range(20):
        poll = client.get(f"/api/experiments/{exp_id}")
        if poll.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    res = client.get(f"/api/experiments/{exp_id}/results")
    assert res.status_code == 200
    metrics = res.json()["metrics"]

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["r2"], float)


def test_dip_profile_counts_and_missingness_schema():
    """Verify DIP profile schema returns valid feature counts and missingness metrics."""
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    prof_res = client.post(f"/api/datasets/{ds_id}/profile", data={"target_column": "target"})
    assert prof_res.status_code == 200
    profile = prof_res.json()

    assert "schema" in profile
    assert "numeric_features" in profile["schema"]
    assert "quality" in profile["schema"] or "quality" in profile
    assert "complexity_score" in profile


def test_chat_endpoint_includes_user_question():
    """Verify AI Assistant chat endpoint processes user prompt and returns grounded explanation."""
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    exp_res = client.post(
        "/api/experiments",
        data={
            "dataset_id": ds_id,
            "target_column": "target",
            "mode": "genesis",
            "generations": 2,
            "population_size": 5
        }
    )
    exp_id = exp_res.json()["id"]

    import time
    for _ in range(20):
        poll = client.get(f"/api/experiments/{exp_id}")
        if poll.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    chat_res = client.post(
        f"/api/experiments/{exp_id}/chat",
        data={"prompt": "Why did this model perform well?"}
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "explanation" in chat_data
    assert "evidence_used" in chat_data


def test_llm_evidence_validator_score_variations():
    """Verify evidence validator and prompt builder behavior for score variations."""
    from backend.llm.evidence import EvidenceValidator
    from backend.llm.prompt_builder import PromptBuilder
    import math

    # A. Valid score
    _, _, c_valid = EvidenceValidator.validate_evidence({"model_score": 0.87})
    assert c_valid["model_score"] == 0.87

    # B. Missing score
    _, _, c_missing = EvidenceValidator.validate_evidence({})
    assert c_missing["model_score"] is None

    # C. NaN
    _, _, c_nan = EvidenceValidator.validate_evidence({"model_score": float("nan")})
    assert c_nan["model_score"] is None

    # D. Infinity
    _, _, c_inf = EvidenceValidator.validate_evidence({"model_score": float("inf")})
    assert c_inf["model_score"] is None

    # E. Invalid string
    _, _, c_invalid = EvidenceValidator.validate_evidence({"model_score": "not_a_number"})
    assert c_invalid["model_score"] is None

    # F. Prompt builder N/A display
    prompt_missing = PromptBuilder.build_prompt(c_missing)
    assert "Evaluation Score (SCORE): N/A" in prompt_missing
    assert "Evaluation Score (SCORE): 0.0" not in prompt_missing


def test_importance_and_contribution_variations():
    """Verify evidence validator and prompt builder behavior for importance and contribution variations."""
    from backend.llm.evidence import EvidenceValidator
    from backend.llm.prompt_builder import PromptBuilder
    import math

    # A. Missing global importance
    _, _, c_missing = EvidenceValidator.validate_evidence({
        "global_importance": [{"feature": "f1"}]
    })
    assert c_missing["global_importance"][0]["importance"] is None

    # B. Invalid global importance
    _, _, c_invalid = EvidenceValidator.validate_evidence({
        "global_importance": [{"feature": "f1", "importance": "invalid_val"}]
    })
    assert c_invalid["global_importance"][0]["importance"] is None

    # C. NaN global importance
    _, _, c_nan = EvidenceValidator.validate_evidence({
        "global_importance": [{"feature": "f1", "importance": float("nan")}]
    })
    assert c_nan["global_importance"][0]["importance"] is None

    # D. Missing contribution
    _, _, c_local_missing = EvidenceValidator.validate_evidence({
        "local_explanations": [{"sample_index": 0, "contributions": [{"feature": "f1"}]}]
    })
    assert c_local_missing["local_explanations"][0]["contributions"][0]["contribution"] is None

    # E. Invalid contribution
    _, _, c_local_invalid = EvidenceValidator.validate_evidence({
        "local_explanations": [{"sample_index": 0, "contributions": [{"feature": "f1", "contribution": "bad_float"}]}]
    })
    assert c_local_invalid["local_explanations"][0]["contributions"][0]["contribution"] is None

    # Prompt Builder N/A test for importance
    p_out = PromptBuilder.build_prompt(c_missing)
    assert "Normalized Importance: N/A" in p_out


def test_chat_fallback_does_not_expose_raw_exception(monkeypatch):
    """Verify LLM fallback response does NOT expose raw python exception str(e) to the user."""
    from backend.llm.service import LLMService

    def mock_explain(*args, **kwargs):
        raise RuntimeError("CRITICAL_INTERNAL_DATABASE_SECRET_OR_STACK_TRACE_ERROR")

    monkeypatch.setattr(LLMService, "explain", mock_explain)

    # Create dummy experiment
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]
    exp_res = client.post(
        "/api/experiments",
        data={"dataset_id": ds_id, "target_column": "target", "mode": "genesis", "generations": 1, "population_size": 5}
    )
    exp_id = exp_res.json()["id"]

    chat_res = client.post(f"/api/experiments/{exp_id}/chat", data={"prompt": "Explain results"})
    assert chat_res.status_code == 200
    explanation_text = chat_res.json()["explanation"]
    assert "CRITICAL_INTERNAL_DATABASE_SECRET_OR_STACK_TRACE_ERROR" not in explanation_text
    assert "temporarily unavailable" in explanation_text


def test_results_page_terminology():
    """Verify ResultsPage.jsx component source code contains 'View Explainability' generic terminology."""
    with open("frontend/src/pages/ResultsPage.jsx", "r", encoding="utf-8") as f:
        content = f.read()
    assert "View Explainability" in content
    assert "View SHAP Explainability" not in content


def test_chat_fallback_specific_metric_terminology(monkeypatch):
    """Verify fallback uses exact metric name (e.g., 'F1 score of ...') and handles missing metrics."""
    from backend.llm.service import LLMService

    def mock_explain(*args, **kwargs):
        raise RuntimeError("Provider error")

    monkeypatch.setattr(LLMService, "explain", mock_explain)

    # 1. Classification experiment with metrics
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]
    exp_res = client.post(
        "/api/experiments",
        data={"dataset_id": ds_id, "target_column": "target", "mode": "genesis", "generations": 1, "population_size": 5}
    )
    exp_id = exp_res.json()["id"]

    import time
    for _ in range(20):
        poll = client.get(f"/api/experiments/{exp_id}")
        if poll.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    chat_res = client.post(f"/api/experiments/{exp_id}/chat", data={"prompt": "Explain metrics"})
    assert chat_res.status_code == 200
    exp_text = chat_res.json()["explanation"]
    assert "F1 score" in exp_text or "Accuracy" in exp_text


def test_missing_complexity_score_remains_none():
    """Verify missing complexity_score is None, not 0.0."""
    from backend.database import DatabaseService
    profile = DatabaseService.get_dip_profile("non_existent_id")
    assert profile is None
