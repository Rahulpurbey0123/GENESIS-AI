"""Unit & integration tests for FastAPI REST API endpoints."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
DATA_DIR = Path("data/test_datasets")


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.1"


def test_dip_endpoint_valid():
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        response = client.post(
            "/dip",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")},
            data={"target_column": "target"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["dip_version"] == "1.1"
    assert data["dataset"]["name"] == "01_numerical_classification.csv"
    assert "binary_features" in data["schema"]
    assert "feature_missingness" in data["quality"]
    assert "target_missingness" in data["quality"]
    assert "complexity_score" in data
    assert 0.0 <= data["complexity_score"] <= 10.0



def test_dip_endpoint_missing_target():
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        response = client.post(
            "/dip",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")},
            data={"target_column": "invalid_target_col"}
        )
    assert response.status_code == 400
    data = response.json()
    assert "Target column 'invalid_target_col' was not found." in data["detail"]


def test_dip_endpoint_non_csv():
    response = client.post(
        "/dip",
        files={"file": ("sample.txt", b"some,text,data", "text/plain")},
        data={"target_column": "data"}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Only CSV files are supported" in data["detail"]


def test_recommend_endpoint_valid():
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        response = client.post(
            "/recommend",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")},
            data={"target_column": "target", "top_k": "5"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["recommendation_version"] == "1.1"
    assert data["task_type"] == "classification"
    assert data["candidate_count_before"] == 10
    assert data["candidate_count_after_filtering"] == 5
    assert data["filtering_reduction"] == 0.50
    assert data["recommended_count"] == 5
    assert data["top_k_selection_ratio"] == 1.0
    assert len(data["recommendations"]) == 5
    assert data["search_space_reduction"] == 0.50


def test_recommend_endpoint_missing_target():
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        response = client.post(
            "/recommend",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")},
            data={"target_column": "non_existent_column"}
        )
    assert response.status_code == 400
    data = response.json()
    assert "Target column 'non_existent_column' was not found" in data["detail"]

