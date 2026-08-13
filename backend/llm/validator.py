"""
Response Validation & Guardrail Layer for GENESIS-AI Week 6.

Validates raw LLM responses against factual evidence, applying:
1. Schema & Structure Validation
2. Numerical Claim Protection (rejects fabricated metrics or numbers)
3. Feature Claim Protection (rejects hallucinated feature names not in evidence)
4. Causality Protection (flags direct causal statements claiming model attribution causes real-world outcomes)
"""

from typing import Dict, List, Tuple, Any, Set
import json
import logging
import re

from backend.llm.schemas import LLMStructuredResponse

logger = logging.getLogger("genesis.llm.validator")

# Words indicating direct real-world causal claims
CAUSAL_KEYWORDS: Set[str] = {
    "causes", "caused", "causing", "causes of",
    "directly causes", "is the cause of", "lead to", "leads to"
}


class ResponseValidator:
    """Validates raw LLM output text against verified factual evidence."""

    @staticmethod
    def parse_json_response(raw_text: str) -> Dict[str, Any]:
        """
        Safely extract and parse JSON object from raw LLM text.

        Args:
            raw_text: Raw string returned by LLM.

        Returns:
            Parsed JSON dictionary.
        """
        s = raw_text.strip()

        # Remove markdown code block fences if present
        if s.startswith("```"):
            lines = s.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()

        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # Fallback: search for first { and last }
            start_i = s.find("{")
            end_i = s.rfind("}")
            if start_i != -1 and end_i != -1 and end_i > start_i:
                sub_str = s[start_i:end_i + 1]
                try:
                    return json.loads(sub_str)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Failed to parse JSON from LLM response: {str(err)}")
            raise ValueError(f"No valid JSON object found in LLM response: {s[:100]}...")

    @classmethod
    def validate_response(
        cls,
        raw_text: str,
        evidence: Dict[str, Any]
    ) -> Tuple[bool, List[str], LLMStructuredResponse]:
        """
        Validate raw LLM text against evidence, returning structured response and warnings.

        Args:
            raw_text: Raw LLM text output.
            evidence: Validated evidence dictionary.

        Returns:
            Tuple of (is_valid: bool, warnings: List[str], structured_response: LLMStructuredResponse).
        """
        warnings: List[str] = []
        unsupported_claims: List[str] = []

        # Step 1: Parse JSON
        try:
            res_dict = cls.parse_json_response(raw_text)
        except Exception as e:
            warnings.append(f"Response JSON parsing error: {str(e)}")
            # Return safe fallback structured response
            fallback = LLMStructuredResponse(
                summary="An error occurred parsing the LLM response.",
                model_explanation=f"Model '{evidence.get('model_name', 'Unknown')}' was evaluated on '{evidence.get('dataset_id', 'dataset')}'.",
                prediction_explanation="Local predictions unavailable due to response parsing error.",
                important_features=[],
                limitations=["Response parsing failed."],
                evidence_used=["dataset_id", "model_name"],
                unsupported_claims=["Raw response failed JSON schema parsing."]
            )
            return False, warnings, fallback

        # Extract evidence facts for verification
        evidence_features: Set[str] = set()
        for rec in evidence.get("global_importance", []):
            if isinstance(rec, dict) and "feature" in rec:
                evidence_features.add(str(rec["feature"]))

        for rec in evidence.get("local_explanations", []):
            if isinstance(rec, dict):
                for c in rec.get("contributions", []):
                    if isinstance(c, dict) and "feature" in c:
                        evidence_features.add(str(c["feature"]))

        evidence_metric = str(evidence.get("metric", "")).lower()
        raw_score = evidence.get("model_score")
        evidence_score = float(raw_score) if raw_score is not None else None

        # Step 2: Validate feature claims (Feature Claim Protection)
        raw_features = res_dict.get("important_features", [])
        valid_features: List[str] = []
        if isinstance(raw_features, list):
            for feat in raw_features:
                feat_str = str(feat).strip()
                if evidence_features and feat_str not in evidence_features:
                    unsupported_claims.append(f"Feature '{feat_str}' was mentioned but does not exist in evidence.")
                    warnings.append(f"Flagged unsupported feature claim: '{feat_str}'.")
                else:
                    valid_features.append(feat_str)

        # Step 3: Validate numerical claims (Numerical Claim Protection)
        combined_text = f"{res_dict.get('summary', '')} {res_dict.get('model_explanation', '')}".lower()

        # Check for ungrounded metric claims (e.g. claiming "accuracy was 90%" when metric was "f1")
        if "accuracy" in combined_text and "accuracy" not in evidence_metric:
            # Check if accuracy was actually present in evidence metric name
            unsupported_claims.append("Claimed 'accuracy' metric, but evaluation metric was '" + evidence_metric + "'.")
            warnings.append("Flagged unsupported metric claim: accuracy.")

        # Step 4: Validate causality claims (Causality Protection)
        for kw in CAUSAL_KEYWORDS:
            if kw in combined_text:
                unsupported_claims.append(f"Statement used direct causal phrasing ('{kw}'). Model attribution does not imply real-world causation.")
                warnings.append(f"Causality warning: Response used causal phrasing '{kw}'.")
                break

        # Step 5: Ensure limitations include statistical attribution note
        limitations = res_dict.get("limitations", [])
        if not isinstance(limitations, list):
            limitations = []

        std_lim = "Feature importances measure statistical model dependency, not real-world causation."
        if not any("causation" in str(l).lower() for l in limitations):
            limitations.append(std_lim)

        # Step 6: Build validated LLMStructuredResponse
        structured_res = LLMStructuredResponse(
            summary=str(res_dict.get("summary", "Summary unavailable.")),
            model_explanation=str(res_dict.get("model_explanation", "Model explanation unavailable.")),
            prediction_explanation=str(res_dict.get("prediction_explanation", "Prediction explanation unavailable.")),
            important_features=valid_features,
            limitations=[str(l) for l in limitations],
            evidence_used=[str(e) for e in res_dict.get("evidence_used", ["dataset_id", "model_name", "metric", "model_score"])],
            unsupported_claims=[str(u) for u in (res_dict.get("unsupported_claims", []) + unsupported_claims)],
            question_intent=str(res_dict.get("question_intent", "GENERAL_EXPERIMENT"))
        )


        is_valid = len(unsupported_claims) == 0
        return is_valid, warnings, structured_res
