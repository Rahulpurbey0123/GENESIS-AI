"""
Main Orchestrator for GENESIS-AI Recommendation Engine v1.0.

Provides deterministic pipeline recommendations derived from Dataset Intelligence Profile (DIP) v1.1.
"""

from typing import Dict, Any, Union, Optional, List, cast
from pathlib import Path
import pandas as pd

from backend.dataset.dip import generate_dip
from backend.recommendation.schemas import (
    RecommendationReport,
    ScoringWeights,
    ThresholdConfig,
    NormalizedDIPSignals
)
from backend.recommendation.registry import PipelineRegistry
from backend.recommendation.normalizer import normalize_dip_signals
from backend.recommendation.filters import apply_compatibility_filters
from backend.recommendation.scorer import compute_pipeline_score
from backend.recommendation.ranker import rank_candidates


class RecommendationEngineError(Exception):
    """Custom exception raised for invalid inputs or execution failures in Recommendation Engine."""
    pass


class RecommendationEngine:
    """
    Deterministic Recommendation Engine v1.0.

    Narrows the candidate pipeline search space based on dataset characteristics
    from Dataset Intelligence Profile (DIP) v1.1 without model fitting or optimization.
    """

    def __init__(
        self,
        registry: Optional[PipelineRegistry] = None,
        weights: Optional[ScoringWeights] = None,
        config: Optional[ThresholdConfig] = None
    ):
        self.registry = registry or PipelineRegistry()
        self.weights = weights or ScoringWeights()
        self.config = config or ThresholdConfig()

    def recommend_from_dip(
        self,
        dip_dict: Dict[str, Any],
        top_k: int = 5
    ) -> RecommendationReport:
        """
        Generate candidate pipeline recommendations from an existing DIP v1.1 profile dictionary.

        Args:
            dip_dict: Structured DIP v1.1 output dictionary.
            top_k: Target number of recommended pipelines (default 5).

        Returns:
            Structured RecommendationReport object.
        """
        if not isinstance(dip_dict, dict) or "target" not in dip_dict:
            raise RecommendationEngineError("Invalid DIP profile structure. Missing 'target' key.")

        if top_k <= 0:
            top_k = 5

        # Step 1: DIP Signal Normalization
        signals: NormalizedDIPSignals = normalize_dip_signals(dip_dict, config=self.config)

        # Step 2: Retrieve All Candidate Pipelines from Registry
        all_candidates = self.registry.get_all_pipelines()
        candidate_count_before = len(all_candidates)

        # Step 3: Stage 1 Compatibility Filtering (Hard Filters)
        compatible_candidates, warnings = apply_compatibility_filters(all_candidates, signals)
        candidate_count_after = len(compatible_candidates)

        if candidate_count_after == 0:
            warnings.append("Zero candidate pipelines passed compatibility filtering.")

        # Step 4: Stage 2 Soft Rule Scoring
        scored_candidates = []
        for candidate in compatible_candidates:
            eval_dict = compute_pipeline_score(
                pipeline=candidate,
                signals=signals,
                weights=self.weights,
                config=self.config
            )
            scored_candidates.append({
                "pipeline": candidate,
                "score": eval_dict["score"],
                "sub_scores": eval_dict["sub_scores"],
                "reasons": eval_dict["reasons"],
            })

        # Step 5: Ranking and Top-K Selection
        recommendations, recommended_count = rank_candidates(scored_candidates, top_k=top_k)

        # Step 6: Separate Filtering Reduction & Top-K Selection Metrics
        if candidate_count_before > 0:
            filtering_reduction = round(1.0 - (candidate_count_after / candidate_count_before), 4)
        else:
            filtering_reduction = 0.0

        if candidate_count_after > 0:
            top_k_selection_ratio = round(recommended_count / candidate_count_after, 4)
        else:
            top_k_selection_ratio = 0.0

        # Backward compatibility alias
        search_space_reduction = filtering_reduction

        # Dataset Summary for Output Report
        dataset_summary = {
            "name": dip_dict.get("dataset", {}).get("name", "dataset.csv"),
            "rows": signals.rows,
            "columns": signals.columns,
            "feature_count": signals.feature_count,
            "complexity_score": signals.complexity_score,
            "complexity_label": signals.complexity_label,
            "dataset_size_category": signals.dataset_size_category,
            "missingness_level": signals.missingness_level,
            "imbalance_severity": signals.imbalance_severity,
            "high_dimensional": signals.high_dimensional_flag,
        }

        # Step 7: Construct Structured Report
        report = RecommendationReport(
            recommendation_version="1.1",
            task_type=signals.task_type,
            candidate_count_before=candidate_count_before,
            candidate_count_after_filtering=candidate_count_after,
            filtering_reduction=filtering_reduction,
            top_k=top_k,
            recommended_count=recommended_count,
            top_k_selection_ratio=top_k_selection_ratio,
            search_space_reduction=search_space_reduction,
            recommendations=recommendations,
            warnings=warnings,
            dataset_summary=dataset_summary,
        )

        return report

    def recommend(
        self,
        file_source_or_df: Union[str, Path, bytes, pd.DataFrame, Dict[str, Any]],
        target_column: Optional[str] = None,
        dataset_name: str = "dataset.csv",
        top_k: int = 5
    ) -> RecommendationReport:
        """
        Convenience wrapper method to generate recommendations from CSV file, bytes, DataFrame, or DIP dict.
        """
        if isinstance(file_source_or_df, dict):
            if "target" in file_source_or_df or "dip_version" in file_source_or_df:
                return self.recommend_from_dip(file_source_or_df, top_k=top_k)
            else:
                raise RecommendationEngineError("Invalid dictionary provided. Expected a valid DIP profile dictionary.")

        if target_column is None:
            raise RecommendationEngineError("target_column must be provided when running recommendation from raw dataset.")

        valid_source = cast(Union[str, Path, bytes, pd.DataFrame], file_source_or_df)
        dip_dict = generate_dip(valid_source, target_column=target_column, dataset_name=dataset_name)
        return self.recommend_from_dip(dip_dict, top_k=top_k)


def recommend_pipelines(
    file_source_or_df: Union[str, Path, bytes, pd.DataFrame, Dict[str, Any]],
    target_column: Optional[str] = None,
    dataset_name: str = "dataset.csv",
    top_k: int = 5,
    weights: Optional[ScoringWeights] = None,
    config: Optional[ThresholdConfig] = None
) -> Dict[str, Any]:
    """
    Public convenience function returning dictionary format of recommendation report.
    """
    engine = RecommendationEngine(weights=weights, config=config)
    report = engine.recommend(
        file_source_or_df=file_source_or_df,
        target_column=target_column,
        dataset_name=dataset_name,
        top_k=top_k
    )
    return report.model_dump()
