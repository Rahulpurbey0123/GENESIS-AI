"""
Regression and Integration Test Suite for AI Assistant Question Intent Routing and Provider Selection.
"""

import os
import json
import pytest

from backend.llm.config import LLMConfig
from backend.llm.client import MockLLMClient, OpenRouterClient, create_llm_client
from backend.llm.prompt_builder import PromptBuilder
from backend.llm.service import LLMService
from backend.llm.schemas import LLMExplanationOutput


@pytest.fixture
def sample_evidence():
    return {
        "dataset_id": "test_dataset.csv",
        "pipeline_id": "classification_random_forest",
        "model_name": "Random Forest Classifier",
        "task_type": "classification",
        "metric": "f1_macro",
        "model_score": 0.8850,
        "method": "permutation_importance",
        "global_importance": [
            {"feature": "age", "importance": 0.4200, "rank": 1},
            {"feature": "income", "importance": 0.3100, "rank": 2},
            {"feature": "education", "importance": 0.1500, "rank": 3}
        ],
        "recommendation_summary": {
            "top_recommendations": [
                {"name": "Random Forest Classifier", "score": 0.92},
                {"name": "Gradient Boosting Classifier", "score": 0.89}
            ],
            "search_space_reduction": 0.65
        },
        "efficiency": {
            "pipelines_evaluated": 20,
            "generations": 5,
            "search_space_reduction": "65.0%"
        }
    }


def test_mock_client_question_intent_routing(sample_evidence):
    """TEST A: Mock client receives different questions and routes to distinct intents."""
    client = MockLLMClient()

    # Question 1: Performance
    p1 = PromptBuilder.build_prompt(sample_evidence, user_prompt="Why did this model perform well?")
    res1_json = json.loads(client.generate(p1))
    assert res1_json["question_intent"] == "PERFORMANCE"

    # Question 2: Feature Importance
    p2 = PromptBuilder.build_prompt(sample_evidence, user_prompt="What are the most important features?")
    res2_json = json.loads(client.generate(p2))
    assert res2_json["question_intent"] == "FEATURE_IMPORTANCE"

    # Question 3: Metric Definition
    p3 = PromptBuilder.build_prompt(sample_evidence, user_prompt="What does F1 macro mean?")
    res3_json = json.loads(client.generate(p3))
    assert res3_json["question_intent"] == "METRIC_DEFINITION"

    # Question 4: Recommendation
    p4 = PromptBuilder.build_prompt(sample_evidence, user_prompt="Why was this model recommended?")
    res4_json = json.loads(client.generate(p4))
    assert res4_json["question_intent"] == "RECOMMENDATION"


def test_question_specific_responses_not_identical(sample_evidence):
    """
    MOST IMPORTANT REGRESSION TEST:
    Send 4 questions and assert that responses are NOT all identical, and each response
    contains evidence relevant to its specific question.
    """
    service = LLMService()

    questions = [
        "Why did this model perform well?",
        "What are the most important features?",
        "What does F1 macro mean?",
        "Why was this model recommended?"
    ]

    outputs = []
    for q in questions:
        out: LLMExplanationOutput = service.explain(
            raw_evidence=sample_evidence,
            mode="technical",
            user_prompt=q
        )
        outputs.append(out)

    summaries = [out.structured_explanation.summary for out in outputs]
    intents = [out.structured_explanation.question_intent for out in outputs]

    # Assert responses are not all identical
    assert len(set(summaries)) == 4, "All 4 responses should be unique and question-specific!"

    # Assert detected intents match questions
    assert intents == ["PERFORMANCE", "FEATURE_IMPORTANCE", "METRIC_DEFINITION", "RECOMMENDATION"]

    # Q1: Performance question contains score/metric evidence
    assert "0.885" in summaries[0] or "f1_macro" in summaries[0].lower() or "score" in summaries[0].lower()

    # Q2: Feature importance question contains top feature evidence ('age')
    assert "age" in summaries[1].lower() or "income" in summaries[1].lower() or "influential" in summaries[1].lower()

    # Q3: Metric definition question contains metric definition concept
    assert "metric" in summaries[2].lower() or "f1" in summaries[2].lower() or "quality" in summaries[2].lower()

    # Q4: Recommendation question contains recommendation evidence
    assert "recommended" in summaries[3].lower() or "dip" in summaries[3].lower() or "rule" in summaries[3].lower()


def test_missing_recommendation_evidence_handling():
    """TEST F: When recommendation evidence is missing, assistant explicitly states it is unavailable."""
    sparse_evidence = {
        "dataset_id": "test_dataset.csv",
        "model_name": "Decision Tree",
        "task_type": "classification",
        "metric": "accuracy",
        "model_score": 0.75,
        "method": "permutation_importance",
        "global_importance": []
    }

    client = MockLLMClient()
    prompt = PromptBuilder.build_prompt(sparse_evidence, user_prompt="Why was this model recommended?")
    res = json.loads(client.generate(prompt))

    assert res["question_intent"] == "RECOMMENDATION"
    assert "unavailable" in res["model_explanation"].lower() or "unavailable" in res["summary"].lower()


def test_openrouter_configuration_and_missing_key_handling():
    """TEST G & H: OpenRouter provider selection and safe error handling when API key is missing."""
    # Test G: Provider selection when openrouter is specified
    cfg = LLMConfig(provider="openrouter", model="google/gemini-2.0-flash-001", api_key=None)
    assert cfg.is_mock() is False
    client = create_llm_client(cfg)
    assert client.provider_name == "openrouter"

    # Test H: Service call with missing key produces grounded fallback without crashing
    service = LLMService(client=client, config=cfg)
    out: LLMExplanationOutput = service.explain(
        raw_evidence={"dataset_id": "test.csv", "model_name": "RF", "metric": "f1", "model_score": 0.90},
        user_prompt="Why did this model perform well?"
    )
    assert out.llm_provider == "openrouter"
    assert out.validation_status == "FAILED"
    assert "API key is missing" in out.warnings[0] or "provider communication error" in out.structured_explanation.summary.lower() or "performance fallback" in out.structured_explanation.summary.lower()
