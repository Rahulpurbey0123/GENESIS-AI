"""
Deterministic Candidate Ranker for Recommendation Engine v1.0.

Ranks compatible candidate pipelines by suitability score descending with deterministic tie-breaking.
Formats output as Top-K RecommendationItem list and computes search space reduction.
"""

from typing import List, Dict, Any, Tuple
from backend.recommendation.schemas import (
    PipelineMetadata,
    RecommendationItem,
    NormalizedDIPSignals
)


def rank_candidates(
    scored_candidates: List[Dict[str, Any]],
    top_k: int = 5
) -> Tuple[List[RecommendationItem], int]:
    """
    Rank candidate pipelines deterministically and select Top-K.

    Tie-breaking rule: If scores are equal, sort alphabetically by pipeline_id ascending.

    Args:
        scored_candidates: List of candidate score dicts containing:
            - "pipeline": PipelineMetadata
            - "score": float
            - "sub_scores": Dict[str, float]
            - "reasons": List[str]
        top_k: Number of top candidate recommendations to return (default 5).

    Returns:
        Tuple of (top_k_recommendation_items, search_space_reduction, recommended_count).
    """
    if top_k <= 0:
        top_k = 5

    # Deterministic sort: Primary key score DESC (-score), Secondary key pipeline_id ASC
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda item: (-item["score"], item["pipeline"].pipeline_id)
    )

    total_compatible = len(sorted_candidates)
    selected = sorted_candidates[:top_k]

    recommendation_items: List[RecommendationItem] = []
    for rank_idx, item in enumerate(selected, start=1):
        p: PipelineMetadata = item["pipeline"]
        recommendation_items.append(
            RecommendationItem(
                rank=rank_idx,
                pipeline_id=p.pipeline_id,
                name=p.name,
                model_family=p.model_family,
                model_name=p.model_name,
                score=item["score"],
                sub_scores=item["sub_scores"],
                reasons=item["reasons"],
                pipeline_metadata=p.model_dump(),
            )
        )

    recommended_count = len(recommendation_items)

    return recommendation_items, recommended_count
