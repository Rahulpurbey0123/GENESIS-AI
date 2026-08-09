"""
Deterministic Rule Engine for Recommendation Engine v1.1.

Evaluates dataset signals against candidate pipeline metadata to derive component suitability
sub-scores and machine-generated heuristic explanations with rule traceability.
"""

from typing import Dict, Any, List, Tuple, Optional
from backend.recommendation.schemas import (
    PipelineMetadata,
    NormalizedDIPSignals,
    ThresholdConfig,
    RecommendationReason
)


class RuleEvaluationResult:
    """Holds sub-scores and structured reasons resulting from rule evaluation for a candidate pipeline."""

    def __init__(self):
        self.sub_scores: Dict[str, float] = {}
        self.reasons: List[RecommendationReason] = []

    def set_sub_score(
        self,
        component: str,
        score: float,
        rule_id: Optional[str] = None,
        reason: Optional[str] = None
    ):
        """Record component suitability sub-score (0.0 to 1.0) and optional structured rule explanation."""
        bounded_score = round(min(max(score, 0.0), 1.0), 4)
        self.sub_scores[component] = bounded_score
        if rule_id and reason:
            self.reasons.append(RecommendationReason(rule_id=rule_id, reason=reason))


def evaluate_dataset_size_rules(
    pipeline: PipelineMetadata,
    signals: NormalizedDIPSignals,
    result: RuleEvaluationResult,
    config: ThresholdConfig
) -> None:
    """Evaluate candidate suitability based on dataset size (rows)."""
    size_cat = signals.dataset_size_category
    family = pipeline.model_family

    if size_cat == "small":
        if family in ("linear", "svm"):
            result.set_sub_score(
                "dataset_size", 0.95,
                rule_id="RULE_SIZE_SMALL_HIGH_SUITABILITY",
                reason=f"{pipeline.name} receives higher suitability under the configured small-dataset heuristic ({signals.rows} rows)."
            )
        elif family in ("tree_ensemble", "knn"):
            result.set_sub_score(
                "dataset_size", 0.90,
                rule_id="RULE_SIZE_SMALL_STANDARD_SUITABILITY",
                reason=f"{pipeline.name} receives standard suitability under the configured small-dataset rule."
            )
        else:
            result.set_sub_score(
                "dataset_size", 0.85,
                rule_id="RULE_SIZE_SMALL_BASE_SUITABILITY",
                reason="Dataset size is compatible with pipeline execution limits."
            )

    elif size_cat == "medium":
        if family == "tree_ensemble":
            result.set_sub_score(
                "dataset_size", 1.0,
                rule_id="RULE_SIZE_MEDIUM_TREE_BOOST",
                reason=f"Tree ensemble candidates receive maximum dataset-size suitability under the medium-dataset rule ({signals.rows} rows)."
            )
        elif family == "linear":
            result.set_sub_score(
                "dataset_size", 0.90,
                rule_id="RULE_SIZE_MEDIUM_LINEAR_SUITABILITY",
                reason="Linear models scale smoothly across medium datasets under configured heuristics."
            )
        elif family == "svm":
            result.set_sub_score(
                "dataset_size", 0.80,
                rule_id="RULE_SIZE_MEDIUM_SVM_MODERATE",
                reason="Kernel SVM receives a moderate suitability rating for medium dataset row counts."
            )
        elif family == "knn":
            result.set_sub_score(
                "dataset_size", 0.75,
                rule_id="RULE_SIZE_MEDIUM_KNN_MODERATE",
                reason="KNN inference overhead receives a lower suitability rating on medium datasets."
            )
        else:
            result.set_sub_score(
                "dataset_size", 0.85,
                rule_id="RULE_SIZE_MEDIUM_BASE_SUITABILITY",
                reason="Dataset size is compatible with pipeline execution limits."
            )

    else:  # "large"
        if pipeline.model_name in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
            result.set_sub_score(
                "dataset_size", 1.0,
                rule_id="RULE_SIZE_LARGE_HGB_BOOST",
                reason=f"Histogram-based gradient boosting receives higher suitability under the large-dataset heuristic ({signals.rows} rows)."
            )
        elif family == "linear":
            result.set_sub_score(
                "dataset_size", 0.95,
                rule_id="RULE_SIZE_LARGE_LINEAR_BOOST",
                reason="Linear algorithms receive higher suitability due to linear O(N) row scaling."
            )
        elif family == "tree_ensemble":
            result.set_sub_score(
                "dataset_size", 0.85,
                rule_id="RULE_SIZE_LARGE_TREE_SUITABILITY",
                reason="Tree ensemble methods receive standard suitability on large datasets."
            )
        elif family in ("svm", "knn"):
            result.set_sub_score(
                "dataset_size", 0.40,
                rule_id="RULE_SIZE_LARGE_HIGH_COST_PENALTY",
                reason=f"{pipeline.name} receives a suitability penalty under the large-dataset complexity rule ({signals.rows} rows)."
            )
        else:
            result.set_sub_score(
                "dataset_size", 0.70,
                rule_id="RULE_SIZE_LARGE_BASE_SUITABILITY",
                reason="Large dataset requires efficient model scaling under configured rules."
            )


def evaluate_feature_type_rules(
    pipeline: PipelineMetadata,
    signals: NormalizedDIPSignals,
    result: RuleEvaluationResult,
    config: ThresholdConfig
) -> None:
    """Evaluate candidate suitability based on feature types (numeric vs categorical)."""
    family = pipeline.model_family

    if signals.numerical_heavy_flag:
        if family in ("linear", "svm"):
            result.set_sub_score(
                "feature_type", 0.95,
                rule_id="RULE_FEAT_NUMERIC_HEAVY_LINEAR_SVM",
                reason=f"Dataset is predominantly numerical ({signals.numeric_ratio * 100:.1f}%), boosting {family} suitability under configured rules."
            )
        elif family in ("tree_ensemble", "knn"):
            result.set_sub_score(
                "feature_type", 0.90,
                rule_id="RULE_FEAT_NUMERIC_HEAVY_STANDARD",
                reason=f"Predominantly numerical feature space ({signals.numeric_ratio * 100:.1f}%) is suitable for candidate execution."
            )
        else:
            result.set_sub_score(
                "feature_type", 0.85,
                rule_id="RULE_FEAT_NUMERIC_BASE",
                reason="Feature type composition is compatible with candidate pipeline."
            )

    elif signals.categorical_heavy_flag:
        if family == "tree_ensemble":
            result.set_sub_score(
                "feature_type", 0.90,
                rule_id="RULE_FEAT_CATEGORICAL_TREE_BOOST",
                reason="Tree models receive higher suitability for categorical-heavy datasets post-encoding."
            )
        elif family == "linear":
            result.set_sub_score(
                "feature_type", 0.75,
                rule_id="RULE_FEAT_CATEGORICAL_LINEAR_ENCODED",
                reason="Categorical encoding expands feature dimensionality, resulting in moderate linear suitability."
            )
        elif family in ("svm", "knn"):
            result.set_sub_score(
                "feature_type", 0.70,
                rule_id="RULE_FEAT_CATEGORICAL_DISTANCE_PENALTY",
                reason="Encoded categorical features increase distance sparsity, resulting in a lower suitability score."
            )
        else:
            result.set_sub_score(
                "feature_type", 0.80,
                rule_id="RULE_FEAT_CATEGORICAL_BASE",
                reason="Feature encoding required for categorical attributes under current pipeline configuration."
            )

    else:
        result.set_sub_score(
            "feature_type", 0.85,
            rule_id="RULE_FEAT_MIXED_STANDARD",
            reason="Mixed numerical and categorical feature composition is handled with standard preprocessing steps."
        )


def evaluate_missingness_rules(
    pipeline: PipelineMetadata,
    signals: NormalizedDIPSignals,
    result: RuleEvaluationResult,
    config: ThresholdConfig
) -> None:
    """Evaluate candidate suitability based on missing data levels."""
    level = signals.missingness_level

    if level == "none":
        result.set_sub_score(
            "missingness", 1.0,
            rule_id="RULE_MISSING_NONE",
            reason="Dataset contains zero missing values, receiving maximum missingness suitability."
        )
    elif level == "low":
        result.set_sub_score(
            "missingness", 0.95,
            rule_id="RULE_MISSING_LOW",
            reason=f"Low missingness rate ({signals.missing_rate * 100:.2f}%) is easily resolved by simple imputation."
        )
    elif level in ("moderate", "high"):
        if pipeline.model_family == "tree_ensemble":
            result.set_sub_score(
                "missingness", 0.90,
                rule_id="RULE_MISSING_TREE_TOLERANCE",
                reason=f"Tree-based models receive higher suitability for imputed missing features ({signals.missing_rate * 100:.1f}% missingness)."
            )
        else:
            result.set_sub_score(
                "missingness", 0.80,
                rule_id="RULE_MISSING_STANDARD_IMPUTE",
                reason=f"Feature missingness ({signals.missing_rate * 100:.1f}%) requires mean/median imputation step."
            )


def evaluate_imbalance_rules(
    pipeline: PipelineMetadata,
    signals: NormalizedDIPSignals,
    result: RuleEvaluationResult,
    config: ThresholdConfig
) -> None:
    """Evaluate candidate suitability based on class imbalance (classification only)."""
    if signals.task_type != "classification":
        result.set_sub_score("imbalance", 1.0)
        return

    severity = signals.imbalance_severity
    if severity == "none":
        result.set_sub_score(
            "imbalance", 1.0,
            rule_id="RULE_IMBALANCE_BALANCED",
            reason="Class distribution is balanced, receiving maximum imbalance suitability."
        )
    elif severity in ("moderate", "severe"):
        if pipeline.supports_class_weight:
            result.set_sub_score(
                "imbalance", 0.95,
                rule_id="RULE_IMBALANCE_CLASS_WEIGHT_BOOST",
                reason=f"Class-weight-capable candidates receive higher suitability under the configured imbalance heuristic (imbalance ratio: {signals.imbalance_ratio:.2f})."
            )
        else:
            result.set_sub_score(
                "imbalance", 0.60,
                rule_id="RULE_IMBALANCE_NO_CLASS_WEIGHT_PENALTY",
                reason=f"Lacks native class-weighting support for imbalanced target (imbalance ratio: {signals.imbalance_ratio:.2f})."
            )


def evaluate_dimensionality_rules(
    pipeline: PipelineMetadata,
    signals: NormalizedDIPSignals,
    result: RuleEvaluationResult,
    config: ThresholdConfig
) -> None:
    """Evaluate candidate suitability based on feature-to-sample dimensionality ratio."""
    if signals.high_dimensional_flag:
        family = pipeline.model_family
        if family == "linear":
            result.set_sub_score(
                "dimensionality", 0.95,
                rule_id="RULE_DIM_HIGH_LINEAR_REGULARIZATION",
                reason=f"Linear models receive higher suitability under the configured high-dimensional heuristic (dimensionality ratio: {signals.dimensionality_ratio:.4f})."
            )
        elif family == "svm":
            result.set_sub_score(
                "dimensionality", 0.90,
                rule_id="RULE_DIM_HIGH_SVM_SUITABILITY",
                reason="SVM receives higher suitability under the configured high-dimensional rule."
            )
        elif family == "tree_ensemble":
            result.set_sub_score(
                "dimensionality", 0.80,
                rule_id="RULE_DIM_HIGH_TREE_MODERATE",
                reason="Tree methods receive moderate suitability in high-dimensional feature spaces."
            )
        elif family == "knn":
            result.set_sub_score(
                "dimensionality", 0.50,
                rule_id="RULE_DIM_HIGH_KNN_PENALTY",
                reason=f"KNN receives a penalty under the high-dimensionality curse rule (dimensionality ratio: {signals.dimensionality_ratio:.4f})."
            )
        else:
            result.set_sub_score(
                "dimensionality", 0.75,
                rule_id="RULE_DIM_HIGH_BASE",
                reason="High dimensionality requires feature regularization under configured heuristics."
            )
    else:
        result.set_sub_score(
            "dimensionality", 0.95,
            rule_id="RULE_DIM_NORMAL_STANDARD",
            reason=f"Dimensionality ratio ({signals.dimensionality_ratio:.4f}) is well within standard limits."
        )


def evaluate_computational_rules(
    pipeline: PipelineMetadata,
    signals: NormalizedDIPSignals,
    result: RuleEvaluationResult,
    config: ThresholdConfig
) -> None:
    """Evaluate computational footprint suitability."""
    cost = pipeline.computational_cost
    size_cat = signals.dataset_size_category

    if cost == "low":
        result.set_sub_score(
            "computational", 1.0,
            rule_id="RULE_COMP_LOW_FOOTPRINT",
            reason="Low computational resource footprint enables rapid pipeline evaluation under configured rules."
        )
    elif cost == "medium":
        if size_cat == "large":
            result.set_sub_score(
                "computational", 0.75,
                rule_id="RULE_COMP_MEDIUM_LARGE_DATA",
                reason="Moderate computational runtime required on large dataset."
            )
        else:
            result.set_sub_score(
                "computational", 0.90,
                rule_id="RULE_COMP_MEDIUM_STANDARD",
                reason="Moderate computational footprint suitable for standard search execution."
            )
    elif cost == "high":
        if size_cat == "large":
            result.set_sub_score(
                "computational", 0.40,
                rule_id="RULE_COMP_HIGH_LARGE_DATA_PENALTY",
                reason="High computational complexity receives a reduced score on large datasets."
            )
        else:
            result.set_sub_score(
                "computational", 0.85,
                rule_id="RULE_COMP_HIGH_SMALL_DATA",
                reason="Higher training complexity offset by small to medium dataset volume."
            )
    else:
        result.set_sub_score(
            "computational", 0.85,
            rule_id="RULE_COMP_BASE",
            reason="Standard computational footprint under configured rules."
        )


def evaluate_all_rules(
    pipeline: PipelineMetadata,
    signals: NormalizedDIPSignals,
    config: Optional[ThresholdConfig] = None
) -> RuleEvaluationResult:
    """
    Run all deterministic recommendation rules for a candidate pipeline.

    Args:
        pipeline: Candidate PipelineMetadata instance.
        signals: NormalizedDIPSignals instance.
        config: Optional ThresholdConfig instance.

    Returns:
        RuleEvaluationResult containing sub-scores dict and structured reasons list.
    """
    if config is None:
        config = ThresholdConfig()

    eval_result = RuleEvaluationResult()

    # Task compatibility sub-score (1.0 for candidate that passed hard filtering)
    eval_result.set_sub_score("task", 1.0)

    # Evaluate components
    evaluate_dataset_size_rules(pipeline, signals, eval_result, config)
    evaluate_feature_type_rules(pipeline, signals, eval_result, config)
    evaluate_missingness_rules(pipeline, signals, eval_result, config)
    evaluate_imbalance_rules(pipeline, signals, eval_result, config)
    evaluate_dimensionality_rules(pipeline, signals, eval_result, config)
    evaluate_computational_rules(pipeline, signals, eval_result, config)

    return eval_result
