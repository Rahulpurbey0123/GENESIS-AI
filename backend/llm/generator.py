"""
High-level LLM Explanation Generator Helpers for GENESIS-AI Week 6.
"""

from typing import Dict, Any, Optional
from backend.llm.config import LLMConfig
from backend.llm.service import LLMService
from backend.llm.schemas import LLMExplanationOutput


def generate_llm_explanation(
    raw_evidence: Any,
    mode: str = "technical",
    config: Optional[LLMConfig] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None
) -> LLMExplanationOutput:
    """
    High-level convenience helper function to generate an evidence-grounded LLM explanation.

    Args:
        raw_evidence: ExplanationOutput object or evidence dictionary.
        mode: Explanation mode ('simple', 'technical', 'prediction', 'research', 'pipeline').
        config: Optional LLMConfig object.
        provider_override: Optional provider override string ('mock' or 'openrouter').
        model_override: Optional model identifier override string.

    Returns:
        Structured LLMExplanationOutput object.
    """
    service = LLMService(config=config)
    return service.explain(
        raw_evidence=raw_evidence,
        mode=mode,
        provider_override=provider_override,
        model_override=model_override
    )
