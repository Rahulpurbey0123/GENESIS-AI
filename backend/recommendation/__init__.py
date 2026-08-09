"""
GENESIS-AI Intelligent Pipeline Recommendation Engine Package v1.0.

Provides deterministic candidate pipeline recommendation derived from Dataset Intelligence Profile (DIP) v1.1.
"""

from backend.recommendation.schemas import (
    PipelineStep,
    PipelineMetadata,
    NormalizedDIPSignals,
    ScoringWeights,
    ThresholdConfig,
    RecommendationItem,
    RecommendationReport,
)
from backend.recommendation.registry import (
    PipelineRegistry,
    CLASSIFICATION_PIPELINES,
    REGRESSION_PIPELINES,
)
from backend.recommendation.normalizer import normalize_dip_signals
from backend.recommendation.filters import apply_compatibility_filters
from backend.recommendation.rules import evaluate_all_rules, RuleEvaluationResult
from backend.recommendation.scorer import compute_pipeline_score
from backend.recommendation.ranker import rank_candidates
from backend.recommendation.engine import (
    RecommendationEngine,
    RecommendationEngineError,
    recommend_pipelines,
)

__all__ = [
    "PipelineStep",
    "PipelineMetadata",
    "NormalizedDIPSignals",
    "ScoringWeights",
    "ThresholdConfig",
    "RecommendationItem",
    "RecommendationReport",
    "PipelineRegistry",
    "CLASSIFICATION_PIPELINES",
    "REGRESSION_PIPELINES",
    "normalize_dip_signals",
    "apply_compatibility_filters",
    "evaluate_all_rules",
    "RuleEvaluationResult",
    "compute_pipeline_score",
    "rank_candidates",
    "RecommendationEngine",
    "RecommendationEngineError",
    "recommend_pipelines",
]
