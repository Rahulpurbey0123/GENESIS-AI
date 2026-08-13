"""
Unit & Integration tests for Week 8 REST API Endpoints, Dataset Ingestion, Optimization Job Runner, & AI Assistant Chat.
"""

import time
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
DATA_DIR = Path("data/test_datasets")


def test_dataset_upload_valid_csv():
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        response = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["name"] == "01_numerical_classification.csv"
    assert data["rows"] > 0
    assert data["columns"] > 1
    assert data["suggested_target"] == "target"


def test_dataset_upload_invalid_format():
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("invalid.txt", b"invalid text data", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Only .csv files are supported" in data["detail"] or "CSV" in data["detail"]


def test_dataset_get_metadata():
    # Upload first
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    response = client.get(f"/api/datasets/{ds_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == ds_id
    assert data["rows"] > 0


def test_dataset_profile_api():
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    profile_res = client.post(
        f"/api/datasets/{ds_id}/profile",
        data={"target_column": "target"}
    )
    assert profile_res.status_code == 200
    data = profile_res.json()
    assert "complexity_score" in data
    assert data["dataset"]["rows"] > 0

    # Test GET profile
    get_res = client.get(f"/api/datasets/{ds_id}/profile")
    assert get_res.status_code == 200
    assert get_res.json()["complexity_score"] == data["complexity_score"]


def test_experiment_flow_e2e():
    # Step 1: Upload dataset
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    # Step 2: Create experiment
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
    exp_data = exp_res.json()
    exp_id = exp_data["id"]
    assert exp_data["status"] == "RUNNING"

    # Step 3: Poll status until COMPLETED (with timeout)
    max_wait = 10
    start = time.time()
    final_status = "RUNNING"
    while time.time() - start < max_wait:
        poll_res = client.get(f"/api/experiments/{exp_id}")
        assert poll_res.status_code == 200
        final_status = poll_res.json()["status"]
        if final_status in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)

    assert final_status == "COMPLETED"

    # Step 4: Get Recommendations
    rec_res = client.get(f"/api/experiments/{exp_id}/recommendations")
    assert rec_res.status_code == 200
    assert len(rec_res.json()["recommendations"]) > 0

    # Step 5: Get Results
    res_res = client.get(f"/api/experiments/{exp_id}/results")
    assert res_res.status_code == 200
    res_data = res_res.json()
    assert "best_pipeline" in res_data
    assert "metrics" in res_data
    assert "accuracy" in res_data["metrics"] or "mae" in res_data["metrics"]

    # Step 6: Get Explanations
    exp_res = client.get(f"/api/experiments/{exp_id}/explanations")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert "shap" in exp_data
    assert "global_importance" in exp_data

    # Step 7: Chat with Grounded AI Assistant
    chat_res = client.post(
        f"/api/experiments/{exp_id}/chat",
        data={"prompt": "Why did this model perform well?"}
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "explanation" in chat_data


def test_list_experiments():
    response = client.get("/api/experiments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_nonexistent_dataset_404():
    response = client.get("/api/datasets/nonexistent_id_9999")
    assert response.status_code == 404


def test_nonexistent_experiment_404():
    response = client.get("/api/experiments/nonexistent_exp_9999")
    assert response.status_code == 404
