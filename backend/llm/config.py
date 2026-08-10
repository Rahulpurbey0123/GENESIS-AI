"""
LLM Layer Configuration for GENESIS-AI Week 6.

Defines provider settings, model parameters, and environment variable loading
with safe defaults so testing works completely offline without API credentials.
"""

import os
from typing import Optional, Any
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration data model for GENESIS-AI LLM interpretation service."""

    provider: str = Field(
        default="mock",
        description="LLM provider: 'mock' for deterministic offline testing, or 'openrouter'."
    )
    model: str = Field(
        default="google/gemini-2.0-flash-001",
        description="Model name/identifier for the provider."
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for real provider calls. Optional for mock mode."
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature for LLM generation (low value for factual consistency)."
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum tokens allowed in LLM response."
    )
    timeout_seconds: float = Field(
        default=15.0,
        description="HTTP request timeout in seconds."
    )
    max_retries: int = Field(
        default=2,
        description="Maximum retry attempts for transient API failures."
    )
    site_url: str = Field(
        default="https://github.com/Rahulpurbey0123/GENESIS-AI",
        description="HTTP Referer header for OpenRouter API analytics."
    )
    site_name: str = Field(
        default="GENESIS-AI AutoML Engine",
        description="X-Title header for OpenRouter API analytics."
    )

    def __init__(self, **data: Any):
        if "provider" not in data:
            data["provider"] = os.getenv("LLM_PROVIDER", "mock").lower()
        if "model" not in data:
            data["model"] = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
        if "api_key" not in data:
            data["api_key"] = os.getenv("OPENROUTER_API_KEY", None)
        super().__init__(**data)

    def is_mock(self) -> bool:
        """Check whether the configuration uses mock provider mode."""
        return str(self.provider).lower() in ("mock", "test", "offline")


def get_default_config() -> LLMConfig:
    """Factory function for default LLMConfig."""
    return LLMConfig()
