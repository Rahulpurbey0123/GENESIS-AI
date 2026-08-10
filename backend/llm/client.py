"""
LLM Provider Abstraction & Clients for GENESIS-AI Week 6.

Defines the LLMClient interface and implementations for:
- MockLLMClient: Deterministic mock provider for offline testing.
- OpenRouterClient: Provider client connecting to OpenRouter API.
"""

from abc import ABC, abstractmethod
import json
import logging
import time
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional

from backend.llm.config import LLMConfig

logger = logging.getLogger("genesis.llm.client")


class LLMClient(ABC):
    """Abstract Base Class for LLM providers."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the LLM provider."""
        pass

    @property
    def model_name(self) -> str:
        """Name of the model being used."""
        return self.config.model

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """
        Generate raw text response from the LLM provider.

        Args:
            prompt: User/evidence prompt string.
            system_instruction: System prompt instructions.

        Returns:
            Raw text/JSON response string.
        """
        pass


class MockLLMClient(LLMClient):
    """
    Deterministic Mock LLM Client for offline testing and benchmarking.
    Generates structured, evidence-grounded JSON explanations without network requests or API keys.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """Generate a deterministic, structured JSON explanation based on prompt evidence."""
        dataset_name = "dataset.csv"
        model_name = "Candidate Pipeline"
        metric = "score"
        score_val = "N/A"
        method_val = "explanation method"
        mode = "technical"
        top_features: List[str] = []

        for line in prompt.split("\n"):
            sline = line.strip()
            if sline.startswith("Dataset:"):
                dataset_name = sline.split("Dataset:")[1].strip()
            elif sline.startswith("Model:"):
                model_name = sline.split("Model:")[1].strip()
            elif sline.startswith("Evaluation Metric:"):
                metric = sline.split("Evaluation Metric:")[1].strip()
            elif sline.startswith("Validation Score:"):
                score_val = sline.split("Validation Score:")[1].strip()
            elif sline.startswith("Explanation Strategy:"):
                method_val = sline.split("Explanation Strategy:")[1].strip()
            elif sline.startswith("EXPLANATION MODE:"):
                mode = sline.split("EXPLANATION MODE:")[1].strip().lower()
            elif sline.startswith("- Feature:"):
                fname = sline.split("Feature:")[1].split(",")[0].strip()
                if fname and fname not in top_features:
                    top_features.append(fname)

        # Mode-specific narrative responses
        if mode == "simple":
            summary = (
                f"GENESIS-AI analyzed '{dataset_name}' and selected the '{model_name}' pipeline, "
                f"achieving a validation {metric} score of {score_val}."
            )
            model_exp = (
                f"The model relies primarily on key features to make its decisions. "
                f"Features were ranked using the {method_val} strategy."
            )
            pred_exp = "Individual sample predictions reflect these top feature influences."
        elif mode == "research":
            summary = (
                f"Empirical evaluation on dataset '{dataset_name}' selected candidate model '{model_name}' "
                f"with validation score {score_val} ({metric})."
            )
            model_exp = (
                f"Post-hoc attribution executed via '{method_val}' strategy without model retraining. "
                f"Ranked feature importances preserve strictly non-negative standardized bounds."
            )
            pred_exp = "Local sample explanations evaluate representative predictions against ground truth targets."
        elif mode == "pipeline":
            summary = (
                f"GENESIS-AI pipeline selection for '{dataset_name}' identified '{model_name}' "
                f"as the top candidate algorithm."
            )
            model_exp = (
                f"Selection was guided by Dataset Intelligence Profile (DIP) compatibility rules "
                f"and evolutionary optimization search."
            )
            pred_exp = "Pipeline selection process evaluated feature compatibility and model complexity bounds."
        elif mode == "prediction":
            summary = (
                f"Representative prediction explanations for dataset '{dataset_name}' "
                f"using model '{model_name}'."
            )
            model_exp = f"Global model structure evaluated via {method_val} strategy."
            pred_exp = (
                f"Local attributions highlight individual feature influences for selected representative samples, "
                f"comparing predictions against actual targets."
            )
        else:
            # technical mode (default)
            summary = (
                f"Technical interpretation for dataset '{dataset_name}': Pipeline '{model_name}' "
                f"achieved a validation score of {score_val} ({metric})."
            )
            model_exp = (
                f"The fitted model was explained using '{method_val}'. "
                f"Global feature rankings reflect empirical model attribution."
            )
            pred_exp = (
                "Local prediction explanations identify sample-level feature contributions "
                "for representative validation instances."
            )

        response_dict = {
            "summary": summary,
            "model_explanation": model_exp,
            "prediction_explanation": pred_exp,
            "important_features": top_features[:5],
            "limitations": [
                "Feature importances measure statistical model dependency, not real-world causation.",
                "Multicollinearity among features can split importance across correlated columns."
            ],
            "evidence_used": ["dataset_id", "task_type", "model_name", "metric", "model_score", "method", "global_importance"],
            "unsupported_claims": []
        }

        return json.dumps(response_dict, indent=2)


class OpenRouterClient(LLMClient):
    """
    OpenRouter API Client connecting to OpenRouter's unified LLM endpoint.
    Requires OPENROUTER_API_KEY environment variable.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """
        Send prompt to OpenRouter API and return raw response string.
        """
        if not self.config.api_key:
            raise ValueError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY environment variable "
                "or use provider='mock' for offline testing."
            )

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.site_url,
            "X-Title": self.config.site_name,
        }

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"}
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.API_URL, data=data_bytes, headers=headers, method="POST")

        attempts = 0
        last_error: Optional[Exception] = None

        while attempts <= self.config.max_retries:
            attempts += 1
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    response_text = response.read().decode("utf-8")
                    res_json = json.loads(response_text)

                    choices = res_json.get("choices", [])
                    if choices and len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                        return content
                    else:
                        raise RuntimeError("OpenRouter API returned empty choices array.")

            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                last_error = RuntimeError(f"OpenRouter HTTP {e.code} error: {err_body}")
                if e.code in (401, 403):
                    raise last_error
                elif e.code in (429, 500, 502, 503, 504):
                    time.sleep(1.0 * attempts)
                else:
                    raise last_error

            except urllib.error.URLError as e:
                last_error = RuntimeError(f"OpenRouter connection error: {str(e.reason)}")
                time.sleep(1.0 * attempts)

            except Exception as e:
                last_error = e
                time.sleep(1.0 * attempts)

        raise RuntimeError(f"OpenRouter API call failed after {self.config.max_retries + 1} attempts: {str(last_error)}")


def create_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Factory function to instantiate appropriate LLMClient based on configuration."""
    cfg = config or LLMConfig()
    if cfg.is_mock():
        return MockLLMClient(cfg)
    else:
        return OpenRouterClient(cfg)
