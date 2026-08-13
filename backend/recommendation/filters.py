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


def apply_compatibility_filters_with_reasons(
    candidates: List[PipelineMetadata],
    signals: NormalizedDIPSignals
) -> Tuple[List[PipelineMetadata], List[str], List[Dict[str, Any]]]:
    """
    Apply all Stage 1 compatibility filters sequentially and track detailed exclusion reasons.

    Args:
        candidates: Initial candidate pipeline list.
        signals: NormalizedDIPSignals instance.

    Returns:
        Tuple of (filtered_candidates, filter_warnings, excluded_candidates).
    """
    warnings: List[str] = []
    excluded_candidates: List[Dict[str, Any]] = []
    target_task = signals.task_type.lower()
    has_categorical = signals.categorical_features > 0

    post_task_candidates: List[PipelineMetadata] = []

    # 1. Task compatibility filter
    for p in candidates:
        if p.task.lower() != target_task:
            excluded_candidates.append({
                "pipeline_id": p.pipeline_id,
                "name": p.name,
                "model_name": p.model_name,
                "task": p.task,
                "reason": f"Target task is '{signals.task_type.upper()}', but candidate model is designed for '{p.task.upper()}' task."
            })
        else:
            post_task_candidates.append(p)

    if not post_task_candidates:
        warnings.append(f"No candidate pipelines match task type '{signals.task_type}'.")

    # 2. Preprocessing compatibility filter
    post_prep_candidates: List[PipelineMetadata] = []
    for pipeline in post_task_candidates:
        if has_categorical and not pipeline.handles_categorical_natively:
            has_encoder = any(step.step_type == "encoder" for step in pipeline.steps)
            if not has_encoder:
                excluded_candidates.append({
                    "pipeline_id": pipeline.pipeline_id,
                    "name": pipeline.name,
                    "model_name": pipeline.model_name,
                    "task": pipeline.task,
                    "reason": f"Dataset contains {signals.categorical_features} categorical feature(s), but '{pipeline.name}' has no categorical encoder step."
                })
                continue
        post_prep_candidates.append(pipeline)

    # 3. Check for target missingness warning
    if signals.target_missing_flag:
        warnings.append(
            f"Target column contains missing values ({signals.target_missingness * 100:.2f}%). "
            "Target missingness must be cleaned before model training."
        )

    return post_prep_candidates, warnings, excluded_candidates


def apply_compatibility_filters(
    candidates: List[PipelineMetadata],
    signals: NormalizedDIPSignals
) -> Tuple[List[PipelineMetadata], List[str]]:
    """
    Apply all Stage 1 compatibility filters sequentially (backward compatible 2-tuple signature).
    """
    post_prep_candidates, warnings, _ = apply_compatibility_filters_with_reasons(candidates, signals)
    return post_prep_candidates, warnings
