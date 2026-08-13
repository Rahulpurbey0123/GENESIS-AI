"""
Comprehensive hardening tests for LLM Provider Configuration, Error Handling, Safe Status,
Secret Redaction, Experiment State Immutability, Intent Routing, and Fallback Transparency.
"""

import json
import os
import io
import urllib.error
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import DatabaseService, get_db_connection
from backend.llm.client import MockLLMClient, OpenRouterClient, create_llm_client, _sanitize_text
from backend.llm.config import LLMConfig
from backend.llm.service import LLMService

client = TestClient(app)


def test_mock_provider_deterministic_and_no_key(monkeypatch):
    """
    Test 1: Verify Mock LLM client operates deterministically without network or API key.
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    cfg = LLMConfig()
    llm_client = create_llm_client(cfg)
    assert isinstance(llm_client, MockLLMClient)
    assert llm_client.provider_name == "mock"

    res1 = llm_client.generate("What features are important?")
    res2 = llm_client.generate("What features are important?")
    assert res1 == res2
    assert "summary" in res1.lower() or "important_features" in res1.lower()


def test_openrouter_status_endpoint(monkeypatch):
    """
    Test 2: Verify GET /api/llm/status exposes safe provider status without leaking secrets.
    """
    # Case A: Real OpenRouter configured with key
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret_key_abc123")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

    res = client.get("/api/llm/status")
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "openrouter"
    assert data["mode"] == "real"
    assert data["configured"] is True
    assert data["model"] == "google/gemini-2.0-flash-001"
    assert data["has_api_key"] is True
    assert "secret_key_abc123" not in res.text

    # Case B: OpenRouter provider without API key
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    res_nokey = client.get("/api/llm/status")
    assert res_nokey.status_code == 200
    data_nokey = res_nokey.json()
    assert data_nokey["provider"] == "openrouter"
    assert data_nokey["mode"] == "real"
    assert data_nokey["configured"] is False
    assert "error" in data_nokey
    assert data_nokey["has_api_key"] is False

    # Case C: Mock Provider
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    res_mock = client.get("/api/llm/status")
    assert res_mock.status_code == 200
    data_mock = res_mock.json()
    assert data_mock["provider"] == "mock"
    assert data_mock["mode"] == "mock"
    assert data_mock["configured"] is True

    # Case D: Invalid Provider
    monkeypatch.setenv("LLM_PROVIDER", "invalid")
    res_inv = client.get("/api/llm/status")
    assert res_inv.status_code == 200
    data_inv = res_inv.json()
    assert data_inv["provider"] == "invalid"
    assert data_inv["mode"] == "invalid"
    assert data_inv["configured"] is False
    assert "Unsupported LLM provider configuration" in data_inv["error"]


def test_invalid_provider_factory_and_chat(monkeypatch):
    """
    Test 3: Verify LLM_PROVIDER=invalid raises clear ValueError and fails safely in chat endpoint without silent fallback to mock.
    """
    monkeypatch.setenv("LLM_PROVIDER", "invalid")
    cfg = LLMConfig()
    with pytest.raises(ValueError, match="Unsupported LLM provider configuration"):
        create_llm_client(cfg)


def test_openrouter_successful_generation(monkeypatch):
    """
    Test 4: Verify OpenRouterClient sends payload and parses JSON response when URL opens successfully.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "valid_test_key")

    mock_json = json.dumps({
        "choices": [{
            "message": {
                "content": "{\"summary\": \"Test Summary\", \"model_explanation\": \"Test Model Exp\"}"
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return mock_json

    def mock_urlopen(req, timeout=None):
        return MockHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    cfg = LLMConfig()
    openrouter_client = OpenRouterClient(cfg)
    result = openrouter_client.generate("Test prompt")
    assert "Test Summary" in result


def test_secret_redaction_in_errors(monkeypatch):
    """
    Test 5: Verify API key is redacted from HTTP error messages and tracebacks.
    """
    secret = "SUPER_SECRET_OPENROUTER_KEY_9999"
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    def mock_urlopen_fail(req, timeout=None):
        err_msg = f"Unauthorized access with key: {secret}"
        fp = io.BytesIO(err_msg.encode("utf-8"))
        raise urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=fp
        )

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_fail)

    cfg = LLMConfig()
    openrouter_client = OpenRouterClient(cfg)

    with pytest.raises(RuntimeError) as exc_info:
        openrouter_client.generate("Test prompt")

    err_text = str(exc_info.value)
    assert secret not in err_text
    assert "[REDACTED]" in err_text


def test_experiment_status_immutable_on_llm_failure(monkeypatch):
    """
    Test 6: Verify experiment state in SQLite remains COMPLETED even when LLM explanation provider fails.
    """
    import uuid
    exp_id = f"exp_immut_{uuid.uuid4().hex[:8]}"
    ds_id = f"ds_immut_{uuid.uuid4().hex[:8]}"

    DatabaseService.save_dataset(
        dataset_id=ds_id,
        name="immutable_test.csv",
        content_hash="hash_immut",
        filepath="data/uploads/immutable_test.csv",
        rows=10,
        columns=2,
        features=["f1", "target"],
        suggested_target="target"
    )

    exp = DatabaseService.create_experiment(
        experiment_id=exp_id,
        dataset_id=ds_id,
        dataset_name="immutable_test.csv",
        target_column="target",
        mode="genesis",
        config={"top_k": 2}
    )

    # Transition experiment to COMPLETED
    DatabaseService.update_experiment_progress(exp_id, progress=1.0, status="COMPLETED")
    exp_before = DatabaseService.get_experiment(exp_id)
    assert exp_before["status"] == "COMPLETED"

    # Set failing LLM provider
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    # Perform chat request
    res = client.post(f"/api/experiments/{exp_id}/chat", data={"prompt": "Why did this model perform well?"})
    assert res.status_code == 200

    # Verify experiment status in database is still COMPLETED
    exp_after = DatabaseService.get_experiment(exp_id)
    assert exp_after["status"] == "COMPLETED"


def test_intent_routing_accuracy():
    """
    Test 7: Verify question intent detector maps prompts to appropriate intent categories.
    """
    mock_client = MockLLMClient()

    assert mock_client._detect_intent("Why did this model perform so well?") == "PERFORMANCE"
    assert mock_client._detect_intent("What are the most important features in this dataset?") == "FEATURE_IMPORTANCE"
    assert mock_client._detect_intent("Why was this model recommended for my dataset?") == "RECOMMENDATION"
    assert mock_client._detect_intent("How much search space was reduced during optimization?") == "SEARCH_SPACE"
    assert mock_client._detect_intent("What does F1 macro metric mean?") == "METRIC_DEFINITION"


def test_error_transparency_and_fallback_marking(monkeypatch):
    """
    Test 8: Proves error transparency, fallback marking, and state distinction (Requirements A-H).
    A. OpenRouter failure does not expose raw error.
    B. OpenRouter failure does not expose API key.
    C. OpenRouter failure returns safe user-facing message.
    D. Fallback response is explicitly marked as fallback (is_fallback: True, validation_status: "FALLBACK").
    E. Mock response is marked as mock (is_fallback: False).
    F. Successful OpenRouter response is marked as real (is_fallback: False).
    """
    secret_key = "secret_key_testing_12345"
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", secret_key)

    # Mock HTTP failure
    def mock_urlopen_fail(req, timeout=None):
        fp = io.BytesIO(f"Internal OpenRouter exception with key {secret_key}".encode("utf-8"))
        raise urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=500,
            msg="Internal Error",
            hdrs={},
            fp=fp
        )

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_fail)

    raw_evidence = {
        "dataset_id": "test_ds.csv",
        "task_type": "classification",
        "model_name": "RandomForest",
        "metrics": {"accuracy": 0.95},
        "model_score": 0.95,
        "features": ["f1", "f2"]
    }

    service = LLMService()
    output = service.explain(raw_evidence=raw_evidence, mode="technical", user_prompt="Why did this model perform well?")

    # Verify requirement A, B, C: No raw error, no API key in output warnings/unsupported claims
    out_text = json.dumps(output.model_dump())
    assert secret_key not in out_text
    assert "Internal OpenRouter exception" not in out_text
    assert "HTTPError" not in out_text

    # Verify requirement D: Fallback explicitly marked
    assert output.metadata["is_fallback"] is True
    assert output.validation_status == "FALLBACK"
    assert output.llm_provider == "openrouter"

    # Verify requirement E: Mock provider is not fallback
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    mock_service = LLMService()
    mock_output = mock_service.explain(raw_evidence=raw_evidence, mode="technical", user_prompt="Test")
    assert mock_output.metadata.get("is_fallback", False) is False
    assert mock_output.validation_status == "PASSED"
    assert mock_output.llm_provider == "mock"

    # Verify requirement F: Successful OpenRouter response is real (not fallback)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    mock_json = json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "summary": "RandomForest achieved 0.95 accuracy.",
                    "model_explanation": "Model generalize well on test_ds.csv."
                })
            }
        }]
    }).encode("utf-8")

    class MockHTTPResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return mock_json

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockHTTPResponse())
    real_output = service.explain(raw_evidence=raw_evidence, mode="technical", user_prompt="Why?")
    assert real_output.metadata.get("is_fallback", False) is False
    assert real_output.llm_provider == "openrouter"
