"""
Explanation Modes for GENESIS-AI Week 6 LLM Explanation Layer.

Defines the 5 supported explanation modes, their target audiences, and mode-specific instructions.
"""

from typing import Dict, List, Set, Any
from enum import Enum


class ExplanationMode(str, Enum):
    """Supported explanation modes for GENESIS-AI LLM interpretation."""

    SIMPLE = "simple"
    TECHNICAL = "technical"
    PREDICTION = "prediction"
    RESEARCH = "research"
    PIPELINE = "pipeline"


VALID_MODES: Set[str] = {mode.value for mode in ExplanationMode}

MODE_DESCRIPTIONS: Dict[str, str] = {
    ExplanationMode.SIMPLE.value: "High-level, plain-language explanation for non-technical users.",
    ExplanationMode.TECHNICAL.value: "Detailed, technical explanation for data science practitioners.",
    ExplanationMode.PREDICTION.value: "Sample-level prediction breakdown based on representative local evidence.",
    ExplanationMode.RESEARCH.value: "Formal research-style methodology explanation with empirical attribution bounds.",
    ExplanationMode.PIPELINE.value: "Automated pipeline selection justification connecting DIP profiling to GA selection."
}

MODE_INSTRUCTIONS: Dict[str, str] = {
    ExplanationMode.SIMPLE.value: (
        "Focus on clear, plain-language summaries without heavy mathematical jargon. "
        "Explain what the model does, which top features influenced the result, "
        "and what the performance score means in simple terms."
    ),
    ExplanationMode.TECHNICAL.value: (
        "Provide a precise, technical interpretation for data science practitioners. "
        "Discuss the explanation strategy used, standardized global feature ranks, "
        "direction indicators, and local contribution vectors where available."
    ),
    ExplanationMode.PREDICTION.value: (
        "Focus specifically on explaining representative sample predictions. "
        "Contrast predictions against ground truth targets for representative cases "
        "(e.g., true positive, false positive, low/high residual) using local feature attributions."
    ),
    ExplanationMode.RESEARCH.value: (
        "Provide a rigorous, research-oriented interpretation of the empirical findings. "
        "Emphasize post-hoc attribution bounds, validation score metrics, "
        "and explicit limitations (attributions are statistical dependencies, not real-world causation)."
    ),
    ExplanationMode.PIPELINE.value: (
        "Explain the AutoML pipeline choice. Detail why this specific model family and preprocessing steps "
        "were selected by GENESIS-AI based on dataset intelligence profile signals and evolutionary optimization."
    )
}


def normalize_mode(mode_str: str) -> str:
    """Normalize input mode string to valid mode value or default to 'technical'."""
    s = str(mode_str).lower().strip()
    if s in VALID_MODES:
        return s
    return ExplanationMode.TECHNICAL.value
