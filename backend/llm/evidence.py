"""
Evidence Extraction & Validation Layer for GENESIS-AI Week 6.

Enforces strict separation between ML computation (facts) and LLM interpretation.
Applies an explicit evidence allowlist, extracts verified facts from Week 5 outputs,
and validates data types, metric names, and finite values before prompt construction.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import logging
import math
import numpy as np
import pandas as pd

logger = logging.getLogger("genesis.llm.evidence")

# Strict allowlist of approved evidence fields
ALLOWED_EVIDENCE_FIELDS: Set[str] = {
    "dataset_id",
    "pipeline_id",
    "model_name",
    "task_type",
    "metric",
    "model_score",
    "method",
    "global_importance",
    "local_explanations",
    "prediction",
    "actual_value",
    "warnings",
    "metadata",
    "recommendation_summary",
    "efficiency",
    "metrics",
    "status",
    "dip_summary",
    "search_space"
}


VALID_TASK_TYPES: Set[str] = {"classification", "regression"}
VALID_METRICS: Set[str] = {"f1", "f1_macro", "accuracy", "rmse", "neg_root_mean_squared_error", "r2", "mae"}
VALID_EXPLANATION_METHODS: Set[str] = {"shap_tree", "linear_coefficients", "native_tree", "permutation_importance", "unsupported"}


class EvidenceExtractor:
    """Extracts approved evidence fields from structured Week 5 ExplanationOutput."""

    @staticmethod
    def extract_evidence(raw_evidence: Any) -> Dict[str, Any]:
        """
        Extract allowed evidence fields from an ExplanationOutput object or dictionary.

        Args:
            raw_evidence: ExplanationOutput instance or dictionary representation.

        Returns:
            Dict containing only approved evidence fields.
        """
        if hasattr(raw_evidence, "model_dump"):
            data_dict = raw_evidence.model_dump()
        elif isinstance(raw_evidence, dict):
            data_dict = raw_evidence
        else:
            raise TypeError(f"Evidence must be ExplanationOutput or dict, got {type(raw_evidence)}")

        extracted: Dict[str, Any] = {}

        # Extract top-level allowed fields
        for field in ALLOWED_EVIDENCE_FIELDS:
            if field in data_dict:
                extracted[field] = data_dict[field]

        # Extract nested full_explanation if present in experiment JSON format
        if "full_explanation" in data_dict and isinstance(data_dict["full_explanation"], dict):
            full_exp = data_dict["full_explanation"]
            for field in ALLOWED_EVIDENCE_FIELDS:
                if field in full_exp and field not in extracted:
                    extracted[field] = full_exp[field]

        return extracted


class EvidenceValidator:
    """Validates extracted evidence data before prompt construction."""

    @staticmethod
    def validate_evidence(evidence: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate evidence schema, data types, finite numbers, and allowed methods.

        Args:
            evidence: Extracted evidence dictionary.

        Returns:
            Tuple of (is_valid: bool, warnings: List[str], cleaned_evidence: Dict[str, Any]).
        """
        warnings: List[str] = []
        cleaned: Dict[str, Any] = {}

        # Check dataset_id
        ds_id = str(evidence.get("dataset_id", "unknown_dataset.csv"))
        cleaned["dataset_id"] = ds_id

        # Check task_type
        task_type = str(evidence.get("task_type", "classification")).lower()
        if task_type not in VALID_TASK_TYPES:
            warnings.append(f"Invalid task_type '{task_type}'. Defaulting to 'classification'.")
            task_type = "classification"
        cleaned["task_type"] = task_type

        # Check model_name
        cleaned["model_name"] = str(evidence.get("model_name", "Unknown Estimator"))
        cleaned["pipeline_id"] = str(evidence.get("pipeline_id", "custom_pipeline"))

        # Check metric & model_score
        metric = str(evidence.get("metric", "score"))
        score_val = evidence.get("model_score")
        if score_val is not None:
            try:
                score_float = float(score_val)
                if math.isnan(score_float) or math.isinf(score_float):
                    warnings.append("Model score contained non-finite float. Excluded score (None).")
                    cleaned["model_score"] = None
                else:
                    cleaned["model_score"] = round(score_float, 4)
            except (ValueError, TypeError):
                warnings.append("Model score could not be parsed to float. Excluded score (None).")
                cleaned["model_score"] = None
        else:
            cleaned["model_score"] = None

        cleaned["metric"] = metric

        # Check method
        method = str(evidence.get("method", "unsupported"))
        if method not in VALID_EXPLANATION_METHODS:
            warnings.append(f"Unrecognized explanation method '{method}'.")
        cleaned["method"] = method

        # Check global_importance records
        raw_global = evidence.get("global_importance", [])
        cleaned_global = []
        if isinstance(raw_global, list):
            for item in raw_global:
                if isinstance(item, dict):
                    fname = str(item.get("feature", "unknown_feature"))
                    imp_val = item.get("importance")
                    imp_clean = None
                    if imp_val is not None:
                        try:
                            imp_float = float(imp_val)
                            if math.isnan(imp_float) or math.isinf(imp_float):
                                warnings.append(f"Non-finite importance value for feature '{fname}'. Excluded importance.")
                                imp_clean = None
                            else:
                                imp_clean = round(imp_float, 4)
                        except (ValueError, TypeError):
                            warnings.append(f"Invalid importance value for feature '{fname}'. Excluded importance.")
                            imp_clean = None

                    rank_val = item.get("rank", len(cleaned_global) + 1)
                    dir_val = item.get("direction", None)

                    cleaned_global.append({
                        "feature": fname,
                        "importance": imp_clean,
                        "rank": int(rank_val) if isinstance(rank_val, int) else len(cleaned_global) + 1,
                        "direction": int(dir_val) if isinstance(dir_val, int) else None
                    })
        cleaned["global_importance"] = cleaned_global

        # Check local_explanations
        raw_local = evidence.get("local_explanations", [])
        cleaned_local = []
        if isinstance(raw_local, list):
            for item in raw_local:
                if isinstance(item, dict):
                    idx_val = item.get("sample_index", 0)
                    cat_name = str(item.get("category", "representative_sample"))
                    pred_val = item.get("prediction", "N/A")
                    act_val = item.get("actual_value", "N/A")
                    base_val = item.get("base_value", None)

                    raw_contribs = item.get("contributions", [])
                    cleaned_contribs = []
                    if isinstance(raw_contribs, list):
                        for c in raw_contribs:
                            if isinstance(c, dict):
                                c_fname = str(c.get("feature", "unknown"))
                                c_fval = c.get("feature_value", None)
                                c_score = c.get("contribution")
                                c_clean = None
                                if c_score is not None:
                                    try:
                                        c_score_float = float(c_score)
                                        if math.isnan(c_score_float) or math.isinf(c_score_float):
                                            warnings.append(f"Non-finite contribution for feature '{c_fname}'. Excluded contribution.")
                                            c_clean = None
                                        else:
                                            c_clean = round(c_score_float, 4)
                                    except (ValueError, TypeError):
                                        warnings.append(f"Invalid contribution for feature '{c_fname}'. Excluded contribution.")
                                        c_clean = None

                                cleaned_contribs.append({
                                    "feature": c_fname,
                                    "feature_value": c_fval,
                                    "contribution": c_clean
                                })

                    cleaned_local.append({
                        "sample_index": int(idx_val) if isinstance(idx_val, int) else 0,
                        "category": cat_name,
                        "prediction": pred_val,
                        "actual_value": act_val,
                        "base_value": round(float(base_val), 4) if isinstance(base_val, (int, float)) and not math.isnan(base_val) else None,
                        "contributions": cleaned_contribs
                    })
        cleaned["local_explanations"] = cleaned_local

        # Check warnings & metadata
        raw_warns = evidence.get("warnings", [])
        cleaned["warnings"] = [str(w) for w in raw_warns] if isinstance(raw_warns, list) else []

        raw_meta = evidence.get("metadata", {})
        cleaned["metadata"] = dict(raw_meta) if isinstance(raw_meta, dict) else {}

        # Preserve structured enrichment fields if present
        for extra_k in ["recommendation_summary", "efficiency", "metrics", "status", "dip_summary", "search_space"]:
            if extra_k in evidence:
                cleaned[extra_k] = evidence[extra_k]

        return True, warnings, cleaned
