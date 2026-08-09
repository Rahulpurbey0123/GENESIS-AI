"""
DIP v1.1 Signal Normalizer for Recommendation Engine v1.0.

Extracts and transforms raw Dataset Intelligence Profile (DIP) v1.1 metrics
into derived recommendation signals without mutating the original DIP dictionary.
"""

from typing import Dict, Any, Optional
from backend.recommendation.schemas import NormalizedDIPSignals, ThresholdConfig


def normalize_dip_signals(
    dip_dict: Dict[str, Any],
    config: Optional[ThresholdConfig] = None
) -> NormalizedDIPSignals:
    """
    Extract and normalize DIP v1.1 metrics into structured recommendation signals.

    Args:
        dip_dict: Structured DIP v1.1 output dictionary.
        config: Optional ThresholdConfig instance.

    Returns:
        NormalizedDIPSignals dataclass instance.
    """
    if config is None:
        config = ThresholdConfig()

    ds = dip_dict.get("dataset", {})
    schema = dip_dict.get("schema", {})
    quality = dip_dict.get("quality", {})
    stats = dip_dict.get("statistics", {})
    target = dip_dict.get("target", {})
    complexity_detail = dip_dict.get("complexity_detail", {})

    rows = ds.get("rows", 0)
    columns = ds.get("columns", 0)
    feature_count = ds.get("feature_count", max(columns - 1, 0))

    numeric_features = schema.get("numeric_features", 0)
    categorical_features = schema.get("categorical_features", 0)
    binary_features = schema.get("binary_features", 0)

    numeric_ratio = schema.get("numeric_ratio", 0.0)
    categorical_ratio = schema.get("categorical_ratio", 0.0)
    binary_ratio = schema.get("binary_ratio", 0.0)

    total_missing = quality.get("total_missing", 0)
    missing_rate = quality.get("missing_rate", 0.0)
    
    raw_feature_missingness = quality.get("feature_missingness", {})
    if isinstance(raw_feature_missingness, dict):
        feature_missingness = raw_feature_missingness.get("missing_rate", missing_rate)
    else:
        feature_missingness = float(raw_feature_missingness)

    raw_target_missingness = quality.get("target_missingness", {})
    if isinstance(raw_target_missingness, dict):
        target_missingness = raw_target_missingness.get("missing_rate", 0.0)
    else:
        target_missingness = float(raw_target_missingness)

    outlier_rate = stats.get("outlier_rate", 0.0)
    mean_skew = stats.get("mean_absolute_skewness", 0.0)

    task_type = target.get("task_type", "classification").lower()
    imbalance_ratio = target.get("imbalance_ratio") or 1.0
    minority_percentage = target.get("minority_percentage")

    complexity_score = dip_dict.get("complexity_score", 0.0)
    complexity_label = complexity_detail.get("label", "Low")

    # 1. Dataset size category
    if rows < config.small_dataset_max_rows:
        dataset_size_category = "small"
    elif rows >= config.large_dataset_min_rows:
        dataset_size_category = "large"
    else:
        dataset_size_category = "medium"

    # 2. Missingness level
    if missing_rate == 0.0:
        missingness_level = "none"
    elif missing_rate < config.moderate_missing_rate:
        missingness_level = "low"
    elif missing_rate < config.high_missing_rate:
        missingness_level = "moderate"
    else:
        missingness_level = "high"

    target_missing_flag = bool(target_missingness > 0.0)

    # 3. Imbalance severity
    if task_type == "classification":
        if imbalance_ratio < config.moderate_imbalance_ratio:
            imbalance_severity = "none"
        elif imbalance_ratio < config.severe_imbalance_ratio:
            imbalance_severity = "moderate"
        else:
            imbalance_severity = "severe"
    else:
        imbalance_severity = "none"

    # 4. Dimensionality ratio & flag
    dimensionality_ratio = round(feature_count / max(rows, 1), 6)
    high_dimensional_flag = (
        dimensionality_ratio > config.high_dimensionality_ratio
        or feature_count > config.high_feature_count
    )

    # 5. Feature composition flags
    categorical_heavy_flag = (categorical_ratio > config.categorical_heavy_ratio)
    numerical_heavy_flag = (numeric_ratio > config.numerical_heavy_ratio)

    return NormalizedDIPSignals(
        task_type=task_type,
        rows=rows,
        columns=columns,
        feature_count=feature_count,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        numeric_ratio=numeric_ratio,
        categorical_ratio=categorical_ratio,
        binary_ratio=binary_ratio,
        total_missing=total_missing,
        missing_rate=missing_rate,
        feature_missingness=feature_missingness,
        target_missingness=target_missingness,
        missingness_level=missingness_level,
        target_missing_flag=target_missing_flag,
        outlier_rate=outlier_rate,
        mean_absolute_skewness=mean_skew,
        imbalance_ratio=imbalance_ratio,
        minority_percentage=minority_percentage,
        imbalance_severity=imbalance_severity,
        dimensionality_ratio=dimensionality_ratio,
        high_dimensional_flag=high_dimensional_flag,
        dataset_size_category=dataset_size_category,
        categorical_heavy_flag=categorical_heavy_flag,
        numerical_heavy_flag=numerical_heavy_flag,
        complexity_score=complexity_score,
        complexity_label=complexity_label,
    )
