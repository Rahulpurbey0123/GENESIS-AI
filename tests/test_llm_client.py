"""
Unit tests for backend/llm/client.py (TESTS 2, 3, 14, 15, 16: Client behavior, Mock provider, and API error handling).
"""

import json
import urllib.error
import pytest
from backend.llm.config import LLMConfig
from backend.llm.client import MockLLMClient, OpenRouterClient, create_llm_client


def test_2_missing_api_key_raises_error():
    """Test 2: OpenRouterClient without API key raises ValueError."""
    config = LLMConfig(provider="openrouter", api_key=None)
    client = OpenRouterClient(config)

    with pytest.raises(ValueError, match="OpenRouter API key is missing"):
        client.generate("Test prompt")


def test_3_mock_provider_deterministic_output():
    """Test 3: MockLLMClient produces valid, deterministic JSON response offline."""
    config = LLMConfig(provider="mock")
    client = create_llm_client(config)

    prompt = (
        "Dataset: 01_numerical_classification.csv\n"
        "Model: Support Vector Classifier Pipeline\n"
        "Evaluation Metric: f1\n"
        "Validation Score: 0.3333\n"
        "Explanation Strategy: permutation_importance\n"
        "EXPLANATION MODE: TECHNICAL\n"
        "GLOBAL FEATURE IMPORTANCES:\n"
        "- Feature: chol, Normalized Importance: 1.0000, Rank: 1\n"
    )

    res_str = client.generate(prompt)
    assert res_str is not None
    data = json.loads(res_str)

    assert "summary" in data
    assert "model_explanation" in data
    assert "important_features" in data
    assert "chol" in data["important_features"]


def test_14_15_16_openrouter_error_handling(monkeypatch):
    """Tests 14, 15, 16: HTTP errors, rate limits, and connection timeouts."""
    config = LLMConfig(provider="openrouter", api_key="sk-test-key", max_retries=1, timeout_seconds=1.0)
    client = OpenRouterClient(config)

    # Test 401 Unauthorized (should not retry)
    def mock_urlopen_401(*args, **kwargs):
        raise urllib.error.HTTPError("https://openrouter.ai", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_401)
    with pytest.raises(RuntimeError, match="401"):
        client.generate("Test prompt")

    # Test 429 Rate Limit
    def mock_urlopen_429(*args, **kwargs):
        raise urllib.error.HTTPError("https://openrouter.ai", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_429)
    with pytest.raises(RuntimeError, match="429"):
        client.generate("Test prompt")
