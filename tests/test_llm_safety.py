"""
Safety Invariant tests for backend/llm (Zero secret leakage & offline test execution).
"""

import os
import json
import pytest
from backend.llm.config import LLMConfig
from backend.llm.service import LLMService


def test_safety_zero_secret_leakage():
    """Verify that generated LLMExplanationOutput metadata never exposes secrets or API keys."""
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-fake-secret-key-1234567890"

    config = LLMConfig(provider="mock")
    service = LLMService(config=config)

    evidence = {
        "dataset_id": "sensitive.csv",
        "model_name": "LogisticRegression",
        "metric": "f1",
        "model_score": 0.88,
        "global_importance": [{"feature": "age", "importance": 1.0, "rank": 1}]
    }

    out = service.explain(evidence, mode="technical")
    out_dict = out.model_dump()
    out_str = json.dumps(out_dict)

    assert "sk-or-v1-fake-secret-key" not in out_str
    assert "sk-or-v1-fake-secret-key" not in str(out.metadata)


def test_safety_offline_test_execution(monkeypatch):
    """Verify that tests execute 100% offline without network sockets or external HTTP calls."""
    # Poison socket creation to ensure offline execution
    import socket
    def mock_socket_disabled(*args, **kwargs):
        raise RuntimeError("Network access attempted during offline test suite execution!")

    monkeypatch.setattr(socket, "socket", mock_socket_disabled)

    config = LLMConfig(provider="mock")
    service = LLMService(config=config)

    evidence = {"dataset_id": "01_test.csv", "model_name": "SVC"}
    out = service.explain(evidence, mode="simple")

    assert out.validation_status in ("PASSED", "PASSED_WITH_WARNINGS")
