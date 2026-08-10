"""
Unit tests for backend/llm/config.py (TEST 1: Configuration).
"""

import os
import pytest
from backend.llm.config import LLMConfig, get_default_config


def test_llm_config_defaults():
    """Test 1A: Verify LLMConfig default values."""
    config = get_default_config()
    assert config.provider in ("mock", "openrouter")
    assert config.model is not None
    assert config.temperature == 0.1
    assert config.max_tokens == 1024
    assert config.timeout_seconds == 15.0


def test_llm_config_is_mock():
    """Test 1B: Verify is_mock helper behavior."""
    cfg_mock = LLMConfig(provider="mock")
    assert cfg_mock.is_mock() is True

    cfg_openrouter = LLMConfig(provider="openrouter", api_key="sk-test")
    assert cfg_openrouter.is_mock() is False


def test_llm_config_env_overrides(monkeypatch):
    """Test 1C: Verify environment variable overrides."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-test-key")

    cfg = LLMConfig()
    assert cfg.provider == "openrouter"
    assert cfg.model == "openai/gpt-4o-mini"
    assert cfg.api_key == "sk-env-test-key"
    assert cfg.is_mock() is False
