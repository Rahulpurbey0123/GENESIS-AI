"""
Integration and hard validation tests for week 8 target isolation and scientific audit updates.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import DatabaseService
from backend.explainability.engine import ExplainabilityEngine
from backend.llm.client import MockLLMClient

client = TestClient(app)
DATA_DIR = Path("data/test_datasets")


def test_target_isolated_dip_profile_queries():
    """Verify that DIP profiles are strictly isolated by (dataset_id, target_column)."""
    # 1. Upload dataset
    csv_path = DATA_DIR / "01_numerical_classification.csv"
    with open(csv_path, "rb") as f:
        upload_res = client.post(
            "/api/datasets/upload",
            files={"file": ("01_numerical_classification.csv", f, "text/csv")}
        )
    ds_id = upload_res.json()["id"]

    # 2. Generate DIP profile for target="target"
    prof_res1 = client.post(f"/api/datasets/{ds_id}/profile", data={"target_column": "target"})
    assert prof_res1.status_code == 200

    # 3. Retrieve DIP profile without specifying target (should default/latest)
    get_res1 = client.get(f"/api/datasets/{ds_id}/profile")
    assert get_res1.status_code == 200
    assert get_res1.json().get("target", {}).get("name") == "target"

    # 4. Retrieve DIP profile specifying target="target"
    get_res2 = client.get(f"/api/datasets/{ds_id}/profile?target_column=target")
    assert get_res2.status_code == 200
    assert get_res2.json().get("target", {}).get("name") == "target"

    # 5. Retrieve DIP profile specifying non-existent target column
    get_res3 = client.get(f"/api/datasets/{ds_id}/profile?target_column=non_existent")
    assert get_res3.status_code == 404


def test_target_leakage_explainability_validation():
    """Verify that ExplainabilityEngine raises ValueError if target column leaks into features."""
    engine = ExplainabilityEngine()
    
    # Create simple dummy dataset
    X = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    y = pd.Series([0, 1, 0], name="feat1") # target has same name as feat1 -> leaks!

    # Create dummy mock estimator
    class MockEstimator:
        def predict(self, X):
            return np.array([0, 1, 0])
        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
        @property
        def feature_importances_(self):
            return np.array([0.7, 0.3])
        @property
        def classes_(self):
            return np.array([0, 1])

    estimator = MockEstimator()

    # Explain should raise ValueError due to target leakage (target name 'feat1' is in features)
    with pytest.raises(ValueError) as excinfo:
        engine.explain(
            pipeline_or_model=estimator,
            X_val=X,
            y_val=y,
            dataset_id="test.csv",
            pipeline_id="leak_pipeline",
            model_name="LeakEstimator",
            task_type="classification",
            metric="accuracy"
        )
    assert "Target leakage detected" in str(excinfo.value)


def test_mock_llm_limitations_and_unsupported_intents():
    """Verify that MockLLMClient identifies LIMITATIONS and UNSUPPORTED questions correctly."""
    mock_client = MockLLMClient()

    # A. Limitations intent detection
    intent_lim = mock_client._detect_intent("What are the boundaries, caveats, or constraints of this model?")
    assert intent_lim == "LIMITATIONS"

    # B. Unsupported intent detection
    intent_unsupported = mock_client._detect_intent("What is the capital city of France?")
    assert intent_unsupported == "UNSUPPORTED"

    # C. Unsupported query output
    output_unsupported = mock_client.generate("What is the capital city of France?")
    import json
    data = json.loads(output_unsupported)
    assert data["question_intent"] == "UNSUPPORTED"
    assert "does not contain enough information to answer that reliably" in data["summary"]


def test_llm_config_endpoint():
    """Verify that the GET /api/config/llm endpoint returns LLM provider and model info."""
    res = client.get("/api/config/llm")
    assert res.status_code == 200
    data = res.json()
    assert "provider" in data
    assert "model" in data
    assert "has_api_key" in data
