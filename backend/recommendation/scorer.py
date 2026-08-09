"""
Deterministic Multi-Criteria Suitability Scorer for Recommendation Engine v1.0.

Calculates weighted suitability scores in [0.0, 1.0] from rule evaluation sub-scores.
"""

from typing import Dict, Any, Optional
from backend.recommendation.schemas import ScoringWeights, PipelineMetadata, NormalizedDIPSignals
from backend.recommendation.rules import evaluate_all_rules, RuleEvaluationResult, ThresholdConfig


def compute_pipeline_score(
    pipeline: PipelineMetadata,
    signals: NormalizedDIPSignals,
    weights: Optional[ScoringWeights] = None,
    config: Optional[ThresholdConfig] = None
) -> Dict[str, Any]:
    """
    Calculate deterministic recommendation suitability score for a candidate pipeline.

    Args:
        pipeline: Candidate PipelineMetadata instance.
        signals: NormalizedDIPSignals instance.
        weights: Optional custom ScoringWeights instance.
        config: Optional custom ThresholdConfig instance.

    Returns:
        Dictionary containing:
        - "score": Overall suitability score in [0.0, 1.0]
        - "sub_scores": Component sub-scores dictionary
        - "reasons": Machine-generated human-readable explanations list
    """
    if weights is None:
        weights = ScoringWeights()

    rule_result: RuleEvaluationResult = evaluate_all_rules(pipeline, signals, config=config)
    sub = rule_result.sub_scores

    # Extract component scores (defaulting to 1.0 if not set)
    s_task = sub.get("task", 1.0)
    s_size = sub.get("dataset_size", 1.0)
    s_feat = sub.get("feature_type", 1.0)
    s_miss = sub.get("missingness", 1.0)
    s_imb = sub.get("imbalance", 1.0)
    s_dim = sub.get("dimensionality", 1.0)
    s_comp = sub.get("computational", 1.0)

    # Compute weighted sum
    total_score = (
        weights.task * s_task +
        weights.dataset_size * s_size +
        weights.feature_type * s_feat +
        weights.missingness * s_miss +
        weights.imbalance * s_imb +
        weights.dimensionality * s_dim +
        weights.computational * s_comp
    )

    bounded_score = round(min(max(total_score, 0.0), 1.0), 4)

    return {
        "score": bounded_score,
        "sub_scores": {
            "task": s_task,
            "dataset_size": s_size,
            "feature_type": s_feat,
            "missingness": s_miss,
            "imbalance": s_imb,
            "dimensionality": s_dim,
            "computational": s_comp,
        },
        "reasons": rule_result.reasons,
    }
