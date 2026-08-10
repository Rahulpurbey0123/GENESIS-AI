"""
Unit tests for backend/llm/validator.py (TESTS 8, 9, 10, 11, 13: Response validation & guardrails).
"""

import json
import pytest
from backend.llm.validator import ResponseValidator


def test_8_structured_response_validation():
    """Test 8: ResponseValidator cleanly parses and validates structured JSON."""
    raw_json = json.dumps({
        "summary": "Valid summary of model performance.",
        "model_explanation": "Model uses age and chol features.",
        "prediction_explanation": "Sample prediction reflects feature attributions.",
        "important_features": ["age", "chol"],
        "limitations": ["Attributions measure model dependency, not real-world causation."],
        "evidence_used": ["dataset_id", "model_name"],
        "unsupported_claims": []
    })

    evidence = {
        "dataset_id": "test.csv",
        "metric": "f1",
        "global_importance": [
            {"feature": "age", "importance": 0.6, "rank": 1},
            {"feature": "chol", "importance": 0.4, "rank": 2}
        ]
    }

    is_valid, warnings, resp = ResponseValidator.validate_response(raw_json, evidence)
    assert is_valid is True
    assert resp.summary == "Valid summary of model performance."
    assert "age" in resp.important_features


def test_9_unsupported_numerical_claim_detection():
    """Test 9: Response claiming 'accuracy' when metric was 'f1' is flagged as unsupported."""
    raw_json = json.dumps({
        "summary": "Model achieved an accuracy of 95 percent on validation.",
        "model_explanation": "Model uses feature f1.",
        "prediction_explanation": "Prediction explanation.",
        "important_features": ["f1"],
        "limitations": [],
        "evidence_used": ["dataset_id"],
        "unsupported_claims": []
    })

    evidence = {
        "dataset_id": "test.csv",
        "metric": "f1_macro",
        "model_score": 0.85,
        "global_importance": [{"feature": "f1", "importance": 1.0, "rank": 1}]
    }

    is_valid, warnings, resp = ResponseValidator.validate_response(raw_json, evidence)
    assert is_valid is False
    assert any("accuracy" in claim.lower() for claim in resp.unsupported_claims)


def test_10_unsupported_feature_claim_detection():
    """Test 10: Response claiming unknown feature ('salary') not in evidence is flagged."""
    raw_json = json.dumps({
        "summary": "Model performance summary.",
        "model_explanation": "Model uses salary and age.",
        "prediction_explanation": "Prediction explanation.",
        "important_features": ["salary", "age"],
        "limitations": [],
        "evidence_used": ["dataset_id"],
        "unsupported_claims": []
    })

    evidence = {
        "dataset_id": "test.csv",
        "metric": "f1",
        "global_importance": [{"feature": "age", "importance": 1.0, "rank": 1}]
    }

    is_valid, warnings, resp = ResponseValidator.validate_response(raw_json, evidence)
    assert is_valid is False
    assert "salary" not in resp.important_features
    assert any("salary" in claim for claim in resp.unsupported_claims)


def test_11_causality_protection():
    """Test 11: Response containing direct causal claims ('feature A causes target B') is flagged."""
    raw_json = json.dumps({
        "summary": "High age causes churn directly.",
        "model_explanation": "Model shows age causes target.",
        "prediction_explanation": "Prediction explanation.",
        "important_features": ["age"],
        "limitations": [],
        "evidence_used": ["dataset_id"],
        "unsupported_claims": []
    })

    evidence = {
        "dataset_id": "test.csv",
        "metric": "f1",
        "global_importance": [{"feature": "age", "importance": 1.0, "rank": 1}]
    }

    is_valid, warnings, resp = ResponseValidator.validate_response(raw_json, evidence)
    assert is_valid is False
    assert any("causal" in claim.lower() for claim in resp.unsupported_claims)


def test_13_malformed_llm_response():
    """Test 13: Malformed non-JSON text produces safe fallback response with status FAILED."""
    raw_text = "This is not JSON text at all!"
    evidence = {"dataset_id": "test.csv", "model_name": "RandomForest"}

    is_valid, warnings, resp = ResponseValidator.validate_response(raw_text, evidence)
    assert is_valid is False
    assert len(warnings) > 0
    assert "failed" in resp.summary.lower() or "error" in resp.summary.lower()
