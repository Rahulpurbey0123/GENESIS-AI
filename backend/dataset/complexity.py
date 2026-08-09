"""
GENESIS DIP Complexity Score calculator module.

Computes a transparent, normalized, configurable engineering heuristic score (0.0 to 10.0)
reflecting dataset difficulty across missingness, outliers, skewness, class imbalance, and dimensionality.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ComplexityWeights(BaseModel):
    """Configurable component weights for GENESIS DIP Complexity Score (must sum to 1.0)."""
    missingness: float = Field(0.20, ge=0.0, le=1.0)
    outliers: float = Field(0.20, ge=0.0, le=1.0)
    skewness: float = Field(0.15, ge=0.0, le=1.0)
    imbalance: float = Field(0.25, ge=0.0, le=1.0)
    dimensionality: float = Field(0.20, ge=0.0, le=1.0)


class ComplexityNormalizationCaps(BaseModel):
    """Normalization saturation caps for raw metric values."""
    max_missing_rate: float = 0.30
    max_outlier_rate: float = 0.15
    max_mean_skewness: float = 3.0
    max_imbalance_ratio: float = 20.0
    max_feature_to_sample_ratio: float = 0.50


def get_complexity_label(score: float) -> str:
    """Categorize 0-10 DIP complexity score into human-readable difficulty label."""
    if score <= 3.0:
        return "Low"
    elif score <= 6.0:
        return "Medium"
    elif score <= 8.0:
        return "High"
    else:
        return "Very High"


def compute_complexity_score(
    profile: Dict[str, Any],
    weights: Optional[ComplexityWeights] = None,
    caps: Optional[ComplexityNormalizationCaps] = None
) -> Dict[str, Any]:
    """
    Calculate the transparent GENESIS DIP Complexity Score.

    Args:
        profile: Computed profile dictionary from profiler module.
        weights: Optional custom weights.
        caps: Optional custom normalization saturation caps.

    Returns:
        Dictionary containing overall score, complexity label, normalized components, weights, and documentation.
    """
    if weights is None:
        weights = ComplexityWeights()
    if caps is None:
        caps = ComplexityNormalizationCaps()

    # 1. Missingness component (M)
    raw_missing_rate = profile["missing"]["missing_rate"]
    norm_missingness = min(raw_missing_rate / caps.max_missing_rate, 1.0)

    # 2. Outlier component (O)
    raw_outlier_rate = profile["outliers"]["outlier_rate"]
    norm_outliers = min(raw_outlier_rate / caps.max_outlier_rate, 1.0)

    # 3. Skewness component (S)
    raw_mean_skew = profile["skewness"]["mean_absolute_skewness"]
    norm_skewness = min(raw_mean_skew / caps.max_mean_skewness, 1.0)

    # 4. Class Imbalance component (I)
    target_info = profile["target"]
    if target_info["task_type"] == "classification":
        raw_imbalance = target_info.get("imbalance_ratio") or 1.0
        # Imbalance ratio ranges from 1.0 (perfectly balanced) upwards
        norm_imbalance = min(max(raw_imbalance - 1.0, 0.0) / (caps.max_imbalance_ratio - 1.0), 1.0)
    else:
        # For regression, imbalance component is 0.0
        raw_imbalance = 1.0
        norm_imbalance = 0.0

    # 5. Dimensionality component (D)
    raw_dim_ratio = profile["dimensionality"]["feature_to_sample_ratio"]
    norm_dimensionality = min(raw_dim_ratio / caps.max_feature_to_sample_ratio, 1.0)

    # Weighted sum
    weighted_sum = (
        weights.missingness * norm_missingness +
        weights.outliers * norm_outliers +
        weights.skewness * norm_skewness +
        weights.imbalance * norm_imbalance +
        weights.dimensionality * norm_dimensionality
    )

    # Scale to [0.0, 10.0]
    score = round(min(max(weighted_sum * 10.0, 0.0), 10.0), 2)
    label = get_complexity_label(score)

    return {
        "score_name": "GENESIS DIP Complexity Score",
        "score": score,
        "complexity_label": label,
        "normalized_components": {
            "missingness": round(norm_missingness, 4),
            "outliers": round(norm_outliers, 4),
            "skewness": round(norm_skewness, 4),
            "imbalance": round(norm_imbalance, 4),
            "dimensionality": round(norm_dimensionality, 4),
        },
        "raw_components": {
            "missing_rate": raw_missing_rate,
            "outlier_rate": raw_outlier_rate,
            "mean_absolute_skewness": raw_mean_skew,
            "imbalance_ratio": raw_imbalance,
            "feature_to_sample_ratio": raw_dim_ratio,
        },
        "weights": weights.model_dump(),
        "caps": caps.model_dump(),
        "disclaimer": "Initial engineering heuristic score (0-10). Higher values indicate higher structural challenge for AutoML search.",
    }
