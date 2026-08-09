"""
Stage 1 Compatibility Filters for Recommendation Engine v1.0.

Hard filters that eliminate candidate pipelines incompatible with the dataset task or structure.
"""

from typing import List, Tuple, Dict, Any
from backend.recommendation.schemas import PipelineMetadata, NormalizedDIPSignals


def filter_task_compatibility(
    candidates: List[PipelineMetadata],
    signals: NormalizedDIPSignals
) -> List[PipelineMetadata]:
    """
    Hard filter: Remove candidate pipelines that do not match the dataset task type.

    Args:
        candidates: List of candidate PipelineMetadata entries.
        signals: NormalizedDIPSignals instance.

    Returns:
        Filtered list of compatible candidate PipelineMetadata entries.
    """
    target_task = signals.task_type.lower()
    return [p for p in candidates if p.task.lower() == target_task]


def filter_preprocessing_compatibility(
    candidates: List[PipelineMetadata],
    signals: NormalizedDIPSignals
) -> List[PipelineMetadata]:
    """
    Hard filter: Validate that candidate pipelines support dataset feature requirements.

    Args:
        candidates: List of candidate PipelineMetadata entries.
        signals: NormalizedDIPSignals instance.

    Returns:
        Filtered list of compatible candidate PipelineMetadata entries.
    """
    compatible = []
    has_categorical = signals.categorical_features > 0

    for pipeline in candidates:
        # If dataset has categorical features, candidate must either handle categorical natively or have an encoder step
        if has_categorical and not pipeline.handles_categorical_natively:
            has_encoder = any(step.step_type == "encoder" for step in pipeline.steps)
            if not has_encoder:
                continue

        compatible.append(pipeline)

    return compatible


def apply_compatibility_filters(
    candidates: List[PipelineMetadata],
    signals: NormalizedDIPSignals
) -> Tuple[List[PipelineMetadata], List[str]]:
    """
    Apply all Stage 1 compatibility filters sequentially.

    Args:
        candidates: Initial candidate pipeline list.
        signals: NormalizedDIPSignals instance.

    Returns:
        Tuple of (filtered_candidates, filter_warnings).
    """
    warnings: List[str] = []

    # 1. Task compatibility filter
    post_task_candidates = filter_task_compatibility(candidates, signals)
    if not post_task_candidates:
        warnings.append(f"No candidate pipelines match task type '{signals.task_type}'.")

    # 2. Preprocessing compatibility filter
    post_prep_candidates = filter_preprocessing_compatibility(post_task_candidates, signals)

    # 3. Check for target missingness warning
    if signals.target_missing_flag:
        warnings.append(
            f"Target column contains missing values ({signals.target_missingness * 100:.2f}%). "
            "Target missingness must be cleaned before model training."
        )

    return post_prep_candidates, warnings
