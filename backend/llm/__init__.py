"""
GENESIS-AI Week 6 Evidence-Grounded LLM Explanation Layer Package.

Provides provider-independent, evidence-grounded LLM interpretations for Week 5 structured outputs.
"""

from backend.llm.config import LLMConfig, get_default_config
from backend.llm.client import LLMClient, MockLLMClient, OpenRouterClient, create_llm_client
from backend.llm.evidence import EvidenceExtractor, EvidenceValidator, ALLOWED_EVIDENCE_FIELDS
from backend.llm.modes import ExplanationMode, VALID_MODES, MODE_DESCRIPTIONS
from backend.llm.prompt_builder import PromptBuilder
from backend.llm.validator import ResponseValidator
from backend.llm.schemas import LLMExplanationRequest, LLMStructuredResponse, LLMExplanationOutput
from backend.llm.service import LLMService
from backend.llm.generator import generate_llm_explanation

__all__ = [
    "LLMConfig",
    "get_default_config",
    "LLMClient",
    "MockLLMClient",
    "OpenRouterClient",
    "create_llm_client",
    "EvidenceExtractor",
    "EvidenceValidator",
    "ALLOWED_EVIDENCE_FIELDS",
    "ExplanationMode",
    "VALID_MODES",
    "MODE_DESCRIPTIONS",
    "PromptBuilder",
    "ResponseValidator",
    "LLMExplanationRequest",
    "LLMStructuredResponse",
    "LLMExplanationOutput",
    "LLMService",
    "generate_llm_explanation",
]
