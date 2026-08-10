"""
Unit tests for backend/llm/modes.py (TEST 17: All 5 explanation modes).
"""

import pytest
from backend.llm.modes import ExplanationMode, VALID_MODES, normalize_mode
from backend.llm.service import LLMService
from backend.llm.config import LLMConfig


def test_17_all_explanation_modes():
    """Test 17: Verify all 5 explanation modes run cleanly and produce valid outputs."""
    evidence = {
        "dataset_id": "05_regression.csv",
        "task_type": "regression",
        "pipeline_id": "regression_linear_regression",
        "model_name": "Linear Regression Pipeline",
        "metric": "rmse",
        "model_score": -8302.48,
        "method": "linear_coefficients",
        "global_importance": [
            {"feature": "square_feet", "importance": 0.7203, "rank": 1, "direction": 1},
            {"feature": "bathrooms", "importance": 0.1901, "rank": 2, "direction": 1}
        ],
        "local_explanations": [
            {
                "sample_index": 0,
                "category": "low_residual",
                "prediction": 413661.78,
                "actual_value": 410000.0,
                "contributions": [
                    {"feature": "square_feet", "feature_value": 0.7795, "contribution": 91589.17}
                ]
            }
        ]
    }

    config = LLMConfig(provider="mock")
    service = LLMService(config=config)

    for mode_enum in ExplanationMode:
        mode_str = mode_enum.value
        out = service.explain(raw_evidence=evidence, mode=mode_str)

        assert out is not None
        assert out.explanation_mode == mode_str
        assert out.validation_status in ("PASSED", "PASSED_WITH_WARNINGS")
        assert out.structured_explanation.summary != ""
        assert out.structured_explanation.model_explanation != ""


def test_normalize_mode_fallback():
    """Test normalize_mode fallback for unknown mode strings."""
    assert normalize_mode("SIMPLE") == "simple"
    assert normalize_mode("technical") == "technical"
    assert normalize_mode("UNKNOWN_MODE") == "technical"
