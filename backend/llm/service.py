"""
Core LLM Explanation Service Coordinator for GENESIS-AI Week 6.

Orchestrates evidence validation, prompt construction, provider execution,
response validation, and structured output formatting. Maintains strict separation
between ML computation (facts) and LLM interpretation.
"""

from typing import Dict, List, Optional, Any, Union
import uuid
import time
import logging

from backend.llm.config import LLMConfig
from backend.llm.client import LLMClient, create_llm_client
from backend.llm.evidence import EvidenceExtractor, EvidenceValidator
from backend.llm.modes import normalize_mode
from backend.llm.prompt_builder import PromptBuilder
from backend.llm.validator import ResponseValidator
from backend.llm.schemas import LLMExplanationRequest, LLMExplanationOutput, LLMStructuredResponse

logger = logging.getLogger("genesis.llm.service")


class LLMService:
    """
    Service layer for evidence-grounded LLM explanations in GENESIS-AI.
    """

    def __init__(self, client: Optional[LLMClient] = None, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = client or create_llm_client(self.config)

    def explain(
        self,
        raw_evidence: Any,
        mode: str = "technical",
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> LLMExplanationOutput:
        """
        Generate an evidence-grounded LLM explanation for Week 5 structured output.

        Args:
            raw_evidence: Week 5 ExplanationOutput instance or dictionary.
            mode: Explanation mode ('simple', 'technical', 'prediction', 'research', 'pipeline').
            provider_override: Optional provider override ('mock' or 'openrouter').
            model_override: Optional model identifier override.
            user_prompt: Optional user question to address specifically.

        Returns:
            Structured LLMExplanationOutput object.
        """
        start_time = time.perf_counter()
        req_id = f"llm_req_{uuid.uuid4().hex[:8]}"
        all_warnings: List[str] = []

        # Determine effective client/config if overrides are specified
        effective_client = self.client
        if provider_override or model_override:
            temp_config = self.config.model_copy(deep=True)
            if provider_override:
                temp_config.provider = provider_override
            if model_override:
                temp_config.model = model_override
            effective_client = create_llm_client(temp_config)

        # Step 1: Extract approved evidence fields using allowlist
        try:
            extracted_evidence = EvidenceExtractor.extract_evidence(raw_evidence)
        except Exception as e:
            all_warnings.append(f"Evidence extraction error: {str(e)}")
            extracted_evidence = {}

        # Step 2: Validate evidence schema & data types
        is_evidence_valid, ev_warnings, cleaned_evidence = EvidenceValidator.validate_evidence(extracted_evidence)
        all_warnings.extend(ev_warnings)

        norm_mode = normalize_mode(mode)

        # Step 3: Build system instruction & controlled user prompt
        system_instruction = PromptBuilder.build_system_instruction()
        full_user_prompt = PromptBuilder.build_prompt(cleaned_evidence, mode=norm_mode, user_prompt=user_prompt)

        # Step 4: Execute LLM provider call
        try:
            raw_llm_response = effective_client.generate(prompt=full_user_prompt, system_instruction=system_instruction)
        except Exception as e:
            from backend.llm.client import _sanitize_text
            api_key = getattr(effective_client.config, "api_key", None)
            clean_err = _sanitize_text(str(e), api_key)
            logger.error(
                f"LLM provider generation failed. Provider: '{effective_client.provider_name}', "
                f"Model: '{effective_client.model_name}', Error: {clean_err}",
                exc_info=True
            )
            if "missing" in clean_err.lower():
                all_warnings.append(f"LLM provider generation error ({effective_client.provider_name}): OpenRouter API key is missing.")
            else:
                all_warnings.append(f"LLM provider generation error ({effective_client.provider_name}): Provider unavailable. Grounded fallback active.")
            # Detect question intent for question-aware fallback
            from backend.llm.client import MockLLMClient
            intent_detector = MockLLMClient()
            intent = intent_detector._detect_intent(user_prompt or "")

            ds_name = cleaned_evidence.get('dataset_id', 'dataset')
            m_name = cleaned_evidence.get('model_name', 'Unknown')
            metric_name = cleaned_evidence.get('metric', 'score')
            score_val = cleaned_evidence.get('model_score', 'N/A')
            top_feats = [r.get("feature") for r in cleaned_evidence.get("global_importance", [])[:3] if isinstance(r, dict)]

            if intent == "PERFORMANCE":
                fallback_summary = f"Performance fallback for '{ds_name}': Model '{m_name}' recorded validation score {score_val} ({metric_name})."
                fallback_model_exp = f"Model '{m_name}' generalization evaluated via metric '{metric_name}'."
            elif intent == "FEATURE_IMPORTANCE":
                fallback_summary = f"Feature attributions fallback for '{ds_name}': Top features include {', '.join(top_feats) if top_feats else 'unspecified'}."
                fallback_model_exp = f"Feature ranking calculated via empirical attributions."
            elif intent == "METRIC_DEFINITION":
                fallback_summary = f"Metric concept fallback: '{metric_name}' measures prediction quality. Stored experiment score is {score_val}."
                fallback_model_exp = f"F1 macro is the unweighted average of per-class F1 scores. Verified result: {score_val}."
            elif intent == "RECOMMENDATION":
                fallback_summary = f"Recommendation fallback for '{ds_name}': Model '{m_name}' was selected via dataset DIP rules."
                fallback_model_exp = f"DIP rules prioritized '{m_name}' based on dataset feature complexity."
            else:
                fallback_summary = f"Grounded fallback for '{ds_name}': Model '{m_name}' achieved score {score_val} ({metric_name})."
                fallback_model_exp = f"Model '{m_name}' evaluated on dataset '{ds_name}'."

            fallback_response = LLMStructuredResponse(
                summary=fallback_summary,
                model_explanation=fallback_model_exp,
                prediction_explanation="Local prediction explanations unavailable due to provider error.",
                important_features=top_feats,
                limitations=["Provider call failed; output contains question-aware fallback evidence summary."],
                evidence_used=["dataset_id", "model_name", "metric", "model_score"],
                unsupported_claims=["LLM provider unavailable; output contains grounded fallback response."],
                question_intent=intent
            )

            end_time = time.perf_counter()
            return LLMExplanationOutput(
                request_id=req_id,
                dataset_id=cleaned_evidence.get("dataset_id", "unknown_dataset.csv"),
                task_type=cleaned_evidence.get("task_type", "classification"),
                model_name=cleaned_evidence.get("model_name", "Unknown Estimator"),
                explanation_mode=norm_mode,
                llm_provider=effective_client.provider_name,
                llm_model=effective_client.model_name,
                structured_explanation=fallback_response,
                validation_status="FALLBACK",
                warnings=all_warnings,
                metadata={
                    "runtime_seconds": round(end_time - start_time, 4),
                    "evidence_fields_count": len(cleaned_evidence),
                    "is_fallback": True
                }
            )

        # Step 5: Validate raw LLM response against evidence
        is_response_valid, resp_warnings, structured_response = ResponseValidator.validate_response(
            raw_text=raw_llm_response,
            evidence=cleaned_evidence
        )
        all_warnings.extend(resp_warnings)

        status = "PASSED"
        if not is_response_valid or len(all_warnings) > 0:
            status = "PASSED_WITH_WARNINGS" if is_response_valid else "FAILED"

        end_time = time.perf_counter()
        elapsed_sec = round(end_time - start_time, 4)

        return LLMExplanationOutput(
            request_id=req_id,
            dataset_id=cleaned_evidence.get("dataset_id", "unknown_dataset.csv"),
            task_type=cleaned_evidence.get("task_type", "classification"),
            model_name=cleaned_evidence.get("model_name", "Unknown Estimator"),
            explanation_mode=norm_mode,
            llm_provider=effective_client.provider_name,
            llm_model=effective_client.model_name,
            structured_explanation=structured_response,
            validation_status=status,
            warnings=all_warnings,
            metadata={
                "runtime_seconds": elapsed_sec,
                "evidence_fields_count": len(cleaned_evidence),
                "is_fallback": False
            }
        )
