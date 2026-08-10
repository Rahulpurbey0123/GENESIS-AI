"""
Unit tests for backend/llm/evidence.py (TESTS 5, 6, 12: Evidence extraction, allowlist, and validation).
"""

import pytest
from backend.llm.evidence import EvidenceExtractor, EvidenceValidator, ALLOWED_EVIDENCE_FIELDS


def test_6_evidence_allowlist_enforcement():
    """Test 6: Verify only approved fields pass through EvidenceExtractor."""
    raw_payload = {
        "dataset_id": "test.csv",
        "task_type": "classification",
        "model_name": "RandomForest",
        "metric": "f1",
        "model_score": 0.95,
        "method": "shap_tree",
        "secret_internal_key": "DO_NOT_EXPOSE",
        "internal_filesystem_path": "/etc/secrets"
    }

    extracted = EvidenceExtractor.extract_evidence(raw_payload)
    assert "dataset_id" in extracted
    assert "model_name" in extracted
    assert "secret_internal_key" not in extracted
    assert "internal_filesystem_path" not in extracted
    assert set(extracted.keys()).issubset(ALLOWED_EVIDENCE_FIELDS)


def test_5_evidence_validation_and_sanitization():
    """Test 5: EvidenceValidator cleans non-finite floats, invalid methods, and metric types."""
    raw_evidence = {
        "dataset_id": "01_test.csv",
        "task_type": "INVALID_TASK_TYPE",
        "model_name": "SVC",
        "metric": "f1_macro",
        "model_score": float("nan"),
        "method": "shap_tree",
        "global_importance": [
            {"feature": "f1", "importance": 0.8, "rank": 1},
            {"feature": "f2", "importance": float("inf"), "rank": 2}
        ]
    }

    is_valid, warnings, cleaned = EvidenceValidator.validate_evidence(raw_evidence)
    assert is_valid is True
    assert len(warnings) > 0
    assert cleaned["task_type"] == "classification"  # Cleaned fallback
    assert cleaned["model_score"] == 0.0              # Cleaned NaN fallback
    assert cleaned["global_importance"][1]["importance"] == 0.0  # Cleaned Inf fallback


def test_12_missing_evidence_handling():
    """Test 12: Missing optional evidence fields are set to explicit defaults without error."""
    raw_evidence = {
        "dataset_id": "sparse_dataset.csv"
    }

    is_valid, warnings, cleaned = EvidenceValidator.validate_evidence(raw_evidence)
    assert is_valid is True
    assert cleaned["dataset_id"] == "sparse_dataset.csv"
    assert cleaned["model_name"] == "Unknown Estimator"
    assert cleaned["global_importance"] == []
    assert cleaned["local_explanations"] == []
