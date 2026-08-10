"""
Integration tests for backend/llm/service.py (LLMService end-to-end flow).
"""

import pytest
from backend.explainability.schemas import ExplanationOutput, FeatureImportanceRecord
from backend.llm.service import LLMService
from backend.llm.config import LLMConfig


def test_llm_service_with_explanation_output_object():
    """Test LLMService directly accepting a Week 5 ExplanationOutput object."""
    exp_output = ExplanationOutput(
        dataset_id="02_categorical_heavy.csv",
        pipeline_id="classification_random_forest",
        model_name="Random Forest Classifier Pipeline",
        task_type="classification",
        metric="f1_macro",
        model_score=1.0,
        method="shap_tree",
        global_importance=[
            FeatureImportanceRecord(feature="department_HR", importance=0.2515, rank=1),
            FeatureImportanceRecord(feature="department_Engineering", importance=0.2368, rank=2)
        ],
        local_explanations=[],
        warnings=[]
    )

    config = LLMConfig(provider="mock")
    service = LLMService(config=config)
    out = service.explain(exp_output, mode="research")

    assert out.dataset_id == "02_categorical_heavy.csv"
    assert out.model_name == "Random Forest Classifier Pipeline"
    assert out.explanation_mode == "research"
    assert out.validation_status in ("PASSED", "PASSED_WITH_WARNINGS")
    assert "department_HR" in out.structured_explanation.important_features


def test_llm_service_provider_override():
    """Test LLMService provider_override parameter."""
    config = LLMConfig(provider="mock")
    service = LLMService(config=config)

    evidence = {"dataset_id": "test.csv", "model_name": "SVC"}
    out = service.explain(evidence, mode="simple", provider_override="mock")

    assert out.llm_provider == "mock"
    assert out.validation_status in ("PASSED", "PASSED_WITH_WARNINGS")
