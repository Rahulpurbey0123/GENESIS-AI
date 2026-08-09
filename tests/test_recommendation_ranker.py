"""
Tests for Candidate Ranker (backend/recommendation/ranker.py).
"""

import pytest
from backend.recommendation.registry import PipelineRegistry
from backend.recommendation.ranker import rank_candidates


def test_ranker_ordering_and_tie_breaking():
    """Verify that ranker sorts descending by score and breaks ties alphabetically by pipeline_id."""
    registry = PipelineRegistry()
    p1 = registry.get_pipeline_by_id("classification_random_forest")
    p2 = registry.get_pipeline_by_id("classification_logistic_regression")
    p3 = registry.get_pipeline_by_id("classification_hist_gradient_boosting")

    scored_candidates = [
        {"pipeline": p1, "score": 0.85, "sub_scores": {}, "reasons": []},
        {"pipeline": p2, "score": 0.95, "sub_scores": {}, "reasons": []},
        {"pipeline": p3, "score": 0.85, "sub_scores": {}, "reasons": []},
    ]

    items, count = rank_candidates(scored_candidates, top_k=3)

    assert count == 3
    assert items[0].pipeline_id == "classification_logistic_regression"
    assert items[0].score == 0.95
    assert items[0].rank == 1

    # Tie-break between p3 ("classification_hist_gradient_boosting") and p1 ("classification_random_forest")
    assert items[1].pipeline_id == "classification_hist_gradient_boosting"
    assert items[1].rank == 2
    assert items[2].pipeline_id == "classification_random_forest"
    assert items[2].rank == 3


def test_ranker_top_k_limiting():
    """Verify Top-K selection limits items correctly."""
    registry = PipelineRegistry()
    clf_pipelines = registry.get_pipelines_by_task("classification")

    scored = [
        {"pipeline": p, "score": 0.80 + (idx * 0.02), "sub_scores": {}, "reasons": []}
        for idx, p in enumerate(clf_pipelines)
    ]

    items, count = rank_candidates(scored, top_k=2)

    assert count == 2
    assert len(items) == 2
    assert items[0].rank == 1
    assert items[1].rank == 2
