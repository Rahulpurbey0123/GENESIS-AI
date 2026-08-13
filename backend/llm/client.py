"""
LLM Provider Abstraction & Clients for GENESIS-AI Week 6.

Defines the LLMClient interface and implementations for:
- MockLLMClient: Deterministic mock provider for offline testing.
- OpenRouterClient: Provider client connecting to OpenRouter API.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
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
    Generates question-aware, evidence-grounded JSON explanations without network requests.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    def _detect_intent(self, user_question: str) -> str:
        q = user_question.lower().strip()

        # Check if the question is unrelated/unsupported by experiment evidence
        keywords = ["limit", "metric", "f1", "rmse", "score", "accuracy", "model", "feature", "pipeline", "search", "reduction", "recommend", "local", "prediction", "explanation", "attributions", "generations", "caveat", "drawback", "boundary", "constraint", "causation"]
        if not q or not any(k in q for k in keywords):
            return "UNSUPPORTED"

        if any(k in q for k in ["why did this model perform", "perform well", "performance", "accuracy", "how good", "score"]):
            return "PERFORMANCE"
        elif any(k in q for k in ["important feature", "most important", "feature importance", "key feature", "top feature", "variables", "drivers"]):
            return "FEATURE_IMPORTANCE"
        elif any(k in q for k in ["f1 macro", "f1 mean", "what does", "metric mean", "meaning of", "explain metric", "what is f1", "what is rmse", "what is accuracy", "what is r2"]):
            return "METRIC_DEFINITION"
        elif any(k in q for k in ["recommended", "why was this model", "recommendation", "selection reason", "dip rule"]):
            return "RECOMMENDATION"
        elif any(k in q for k in ["search space", "reduction", "evaluated", "pipelines evaluated", "generations"]):
            return "SEARCH_SPACE"
        elif any(k in q for k in ["pipeline", "hyperparameter", "architecture", "algorithm"]):
            return "PIPELINE"
        elif any(k in q for k in ["prediction", "sample", "row", "local"]):
            return "PREDICTION"
        elif any(k in q for k in ["limit", "caveat", "drawback", "boundary", "constraint", "causation"]):
            return "LIMITATIONS"
        else:
            return "GENERAL_EXPERIMENT"

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """Generate a question-aware, deterministic structured JSON explanation based on prompt evidence."""
        dataset_name = "dataset.csv"
        model_name = "Candidate Pipeline"
        metric = "score"
        score_val = "N/A"
        method_val = "explanation method"
        mode = "technical"
        top_features: List[str] = []
        experiment_id = "N/A"
        target_column = "N/A"
        opt_mode = "GENESIS"
        user_question = ""
        has_rec_evidence = False

        in_question = False
        for line in prompt.split("\n"):
            sline = line.strip()
            if sline.startswith("USER SPECIFIC QUESTION TO ANSWER"):
                in_question = True
                continue
            elif in_question and sline.startswith("Instructions for User Question:"):
                in_question = False
            elif in_question and sline.startswith('"') and sline.endswith('"'):
                user_question = sline.strip('"')
            elif sline.startswith("Experiment ID:"):
                experiment_id = sline.split("Experiment ID:")[1].strip()
            elif sline.startswith("Dataset:"):
                dataset_name = sline.split("Dataset:")[1].strip()
            elif sline.startswith("Target Column:"):
                target_column = sline.split("Target Column:")[1].strip()
            elif sline.startswith("Optimization Mode:"):
                opt_mode = sline.split("Optimization Mode:")[1].strip()
            elif sline.startswith("Model:"):
                model_name = sline.split("Model:")[1].strip()
            elif sline.startswith("Evaluation Metric:") or sline.startswith("Evaluation Metrics:"):
                metric = sline.split(":")[1].strip()
            elif sline.startswith("Evaluation Score") or sline.startswith("Validation Score:"):
                score_val = sline.split(":")[1].strip()
            elif sline.startswith("Explanation Strategy:"):
                method_val = sline.split("Explanation Strategy:")[1].strip()
            elif sline.startswith("EXPLANATION MODE:"):
                mode = sline.split("EXPLANATION MODE:")[1].strip().lower()
            elif sline.startswith("- Feature:"):
                fname = sline.split("Feature:")[1].split(",")[0].strip()
                if fname and fname not in top_features:
                    top_features.append(fname)
            elif "Recommended Candidate:" in sline:
                has_rec_evidence = True

        intent = self._detect_intent(user_question)

        # Question-aware narrative responses based on intent
        if intent == "PERFORMANCE":
            summary = (
                f"Model '{model_name}' (Experiment {experiment_id}) achieved strong predictive validation on dataset '{dataset_name}' "
                f"for target '{target_column}' with evaluation metric(s): {metric}."
            )
            model_exp = (
                f"Performance reflects robust generalization across cross-validation splits under mode {opt_mode}. "
                f"Key decision patterns were driven by top features: {', '.join(top_features[:3]) if top_features else 'primary feature inputs'}."
            )
            pred_exp = "Evaluation metrics validate individual sample predictions against ground-truth target values."

        elif intent == "FEATURE_IMPORTANCE":
            top_str = ", ".join([f"'{f}'" for f in top_features[:3]]) if top_features else "unspecified features"
            summary = f"The most influential feature attributions for dataset '{dataset_name}' (target '{target_column}') are {top_str}."
            model_exp = (
                f"Global feature rankings derived via '{method_val}' identify '{top_features[0] if top_features else 'the primary feature'}' "
                f"as having the highest relative influence on model output in Experiment {experiment_id}."
            )
            pred_exp = "Local sample explanations reflect how top features push individual predictions away from baseline."

        elif intent == "METRIC_DEFINITION":
            summary = (
                f"Evaluation metrics '{metric}' measure predictive model quality for target '{target_column}' in Experiment {experiment_id}."
            )
            model_exp = (
                f"Metric Definition: {metric} evaluates prediction error/accuracy against true target values. "
                f"Your experiment '{experiment_id}' stored verified result metrics: {metric}."
            )
            pred_exp = "Metric definition concepts apply across all sample predictions in the evaluation split."

        elif intent == "RECOMMENDATION":
            if has_rec_evidence:
                summary = f"Pipeline '{model_name}' was recommended for dataset '{dataset_name}' (target '{target_column}') by DIP profiling rules."
                model_exp = f"DIP dataset rules filtered incompatible model families and prioritized '{model_name}' under mode {opt_mode}."
            else:
                summary = f"Recommendation information for pipeline '{model_name}' on dataset '{dataset_name}'."
                model_exp = f"Explicit recommendation candidate ranking facts are unavailable in the stored evidence payload for experiment '{experiment_id}'."
            pred_exp = "Pipeline recommendation rules evaluate dataset-level characteristics rather than individual sample predictions."

        elif intent == "SEARCH_SPACE":
            summary = f"Search space optimization for Experiment {experiment_id} evaluated candidate pipelines for target '{target_column}'."
            model_exp = f"Evolutionary optimization in {opt_mode} mode pruned search space complexity to select pipeline '{model_name}'."
            pred_exp = "Search space reduction focuses on global pipeline selection."

        elif intent == "PREDICTION":
            summary = f"Local prediction explanations for dataset '{dataset_name}' (target '{target_column}') using model '{model_name}'."
            model_exp = f"Global model attributions evaluated via strategy '{method_val}' for Experiment {experiment_id}."
            pred_exp = "Local attributions highlight sample-level feature contributions for representative validation instances."

        elif intent == "LIMITATIONS":
            summary = f"The analysis of pipeline '{model_name}' on dataset '{dataset_name}' (target '{target_column}') has statistical limitations."
            model_exp = (
                f"Limitations: Feature importances measure statistical model dependency, not real-world causation. "
                f"Multicollinearity among features can split importance across correlated columns. "
                f"Evaluation metrics ({metric}) are specific to the validation split for Experiment {experiment_id}."
            )
            pred_exp = "Local explanations show sample attributions within model boundaries but do not represent physical causality."

        elif intent == "UNSUPPORTED":
            summary = f"The stored experiment evidence for Experiment {experiment_id} does not contain enough information to answer that reliably."
            model_exp = "The user question is outside the scope of the stored AutoML experiment facts (such as model metrics, feature importances, and recommendations)."
            pred_exp = "No prediction or sample validation facts are available for this topic."

        else: # GENERAL_EXPERIMENT or default mode fallback
            if mode == "simple":
                summary = f"GENESIS-AI analyzed '{dataset_name}' for target '{target_column}' (Experiment {experiment_id}) and selected '{model_name}'."
                model_exp = f"The model relies primarily on key features. Features were ranked using '{method_val}'."
                pred_exp = "Individual sample predictions reflect top feature influences."
            else:
                summary = f"Technical interpretation for dataset '{dataset_name}' (Experiment {experiment_id}): Pipeline '{model_name}' achieved metric(s) {metric}."
                model_exp = f"The fitted model was explained using '{method_val}' in {opt_mode} mode."
                pred_exp = "Local prediction explanations identify sample-level feature contributions."

        response_dict = {
            "summary": summary,
            "model_explanation": model_exp,
            "prediction_explanation": pred_exp,
            "important_features": top_features[:5],
            "limitations": [
                "Feature importances measure statistical model dependency, not real-world causation.",
                "Multicollinearity among features can split importance across correlated columns."
            ],
            "evidence_used": ["experiment_id", "dataset_id", "dataset_name", "target_column", "mode", "task_type", "model_name", "metric", "model_score", "method", "global_importance"],
            "unsupported_claims": [],
            "question_intent": intent
        }

        return json.dumps(response_dict, indent=2)



def _sanitize_text(text: str, secret: Optional[str] = None) -> str:
    if not text:
        return text
    if secret and secret in text:
        text = text.replace(secret, "[REDACTED]")
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key and len(env_key) > 3 and env_key in text:
        text = text.replace(env_key, "[REDACTED]")
    return text


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
                    try:
                        res_json = json.loads(response_text)
                    except json.JSONDecodeError as je:
                        raise RuntimeError(f"OpenRouter returned malformed JSON response: {str(je)}")

                    choices = res_json.get("choices", [])
                    if choices and len(choices) > 0:
                        content = choices[0].get("message", {}).get("content", "")
                        if not content:
                            raise RuntimeError("OpenRouter API returned empty response content.")
                        return content
                    else:
                        raise RuntimeError("OpenRouter API returned empty choices array.")

            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                clean_body = _sanitize_text(err_body, self.config.api_key)
                last_error = RuntimeError(f"OpenRouter HTTP {e.code} error: {clean_body}")
                if e.code in (401, 403):
                    raise last_error
                elif e.code in (429, 500, 502, 503, 504):
                    time.sleep(0.1 * attempts)
                else:
                    raise last_error

            except urllib.error.URLError as e:
                clean_reason = _sanitize_text(str(e.reason), self.config.api_key)
                last_error = RuntimeError(f"OpenRouter connection error: {clean_reason}")
                time.sleep(0.1 * attempts)

            except Exception as e:
                clean_msg = _sanitize_text(str(e), self.config.api_key)
                last_error = RuntimeError(f"OpenRouter request error: {clean_msg}")
                time.sleep(0.1 * attempts)

        clean_last = _sanitize_text(str(last_error), self.config.api_key)
        raise RuntimeError(f"OpenRouter API call failed after {self.config.max_retries + 1} attempts: {clean_last}")


def create_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Factory function to instantiate appropriate LLMClient based on configuration."""
    cfg = config or LLMConfig()
    p = str(cfg.provider).lower()
    if p in ("mock", "test", "offline"):
        return MockLLMClient(cfg)
    elif p == "openrouter":
        return OpenRouterClient(cfg)
    else:
        raise ValueError(
            f"Unsupported LLM provider configuration: '{cfg.provider}'. Supported providers are 'mock' and 'openrouter'."
        )
