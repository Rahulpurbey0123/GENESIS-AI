"""
Pydantic Schemas for GENESIS-AI Week 6 LLM Explanation Layer.

Defines structured data contracts for requests, structured LLM responses,
and safe explanation outputs.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class LLMExplanationRequest(BaseModel):
    """Input payload for generating an evidence-grounded LLM explanation."""

    evidence: Dict[str, Any] = Field(
        description="Structured Week 5 ExplanationOutput dictionary or evidence mapping."
    )
    mode: str = Field(
        default="technical",
        description="Explanation mode: 'simple', 'technical', 'prediction', 'research', or 'pipeline'."
    )
    provider_override: Optional[str] = Field(
        default=None,
        description="Optional provider override ('mock' or 'openrouter')."
    )
    model_override: Optional[str] = Field(
        default=None,
        description="Optional model identifier override."
    )


class LLMStructuredResponse(BaseModel):
    """Structured fields returned by the LLM response interpretation layer."""

    summary: str = Field(
        description="High-level evidence-grounded summary."
    )
    model_explanation: str = Field(
        description="Interpretation of selected candidate model and global feature importances."
    )
    prediction_explanation: str = Field(
        default="No local prediction explanations were generated for this method.",
        description="Interpretation of local prediction explanations when available."
    )
    important_features: List[str] = Field(
        default_factory=list,
        description="List of feature names explicitly discussed in the explanation."
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Methodological limitations and cautionary notes."
    )
    evidence_used: List[str] = Field(
        default_factory=list,
        description="List of verified evidence fields utilized in the explanation."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of flagged or rejected claims not supported by evidence."
    )
    question_intent: Optional[str] = Field(
        default="GENERAL_EXPERIMENT",
        description="Detected user question intent (PERFORMANCE, FEATURE_IMPORTANCE, METRIC_DEFINITION, RECOMMENDATION, PIPELINE, PREDICTION, SEARCH_SPACE, GENERAL_EXPERIMENT)."
    )



class LLMExplanationOutput(BaseModel):
    """Complete, validated structured response returned by the LLM explanation service."""

    request_id: str = Field(
        description="Unique request identifier."
    )
    dataset_id: str = Field(
        description="Dataset identifier."
    )
    task_type: str = Field(
        description="Task type: 'classification' or 'regression'."
    )
    model_name: str = Field(
        description="Model estimator name."
    )
    explanation_mode: str = Field(
        description="Explanation mode executed ('simple', 'technical', 'prediction', 'research', 'pipeline')."
    )
    llm_provider: str = Field(
        description="LLM provider used ('mock' or 'openrouter')."
    )
    llm_model: str = Field(
        description="LLM model identifier used."
    )
    structured_explanation: LLMStructuredResponse = Field(
        description="Structured LLM response fields."
    )
    validation_status: str = Field(
        description="Validation status: 'PASSED', 'PASSED_WITH_WARNINGS', or 'FAILED'."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during evidence processing or response validation."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Technical execution metadata (runtime_seconds, prompt_tokens, evidence_count)."
    )
