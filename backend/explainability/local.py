"""
Local Prediction Explanation Module for GENESIS-AI Week 5 Explainability Engine.

Selects up to 5 representative samples based on model predictions and ground truth residuals,
and constructs local prediction explanations with ranked feature contributions.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd

from backend.explainability.schemas import LocalExplanationRecord, FeatureContribution


def select_representative_samples_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    max_samples: int = 5
) -> List[Tuple[int, str]]:
    """
    Select up to 5 representative sample indices for classification tasks:
    - correct_positive (TP)
    - correct_negative (TN)
    - false_positive (FP)
    - false_negative (FN)
    - representative_sample (first available unused sample after higher-priority error categories)

    Args:
        y_true: Ground truth target Series.
        y_pred: Predicted class array.
        max_samples: Maximum number of representative samples to return.

    Returns:
        List of tuples: (original_index: int, category_name: str).
    """
    selected: List[Tuple[int, str]] = []
    used_indices = set()

    y_t = np.array(y_true)
    indices = np.arange(len(y_t))

    # Identify categories
    tp_mask = (y_t == 1) & (y_pred == 1)
    tn_mask = (y_t == 0) & (y_pred == 0)
    fp_mask = (y_t == 0) & (y_pred == 1)
    fn_mask = (y_t == 1) & (y_pred == 0)

    categories = [
        ("correct_positive", tp_mask),
        ("correct_negative", tn_mask),
        ("false_positive", fp_mask),
        ("false_negative", fn_mask),
    ]

    for cat_name, mask in categories:
        matching_indices = indices[mask]
        for idx in matching_indices:
            if idx not in used_indices:
                selected.append((int(idx), cat_name))
                used_indices.add(idx)
                break
        if len(selected) >= max_samples:
            return selected

    # Fill remaining slots with representative unselected samples
    for idx in indices:
        if idx not in used_indices:
            selected.append((int(idx), "representative_sample"))
            used_indices.add(idx)
            if len(selected) >= max_samples:
                break

    return selected[:max_samples]


def select_representative_samples_regression(
    y_true: pd.Series,
    y_pred: np.ndarray,
    max_samples: int = 5
) -> List[Tuple[int, str]]:
    """
    Select up to 5 representative sample indices for regression tasks based on residuals:
    - low_residual (smallest absolute error)
    - high_residual (largest absolute error)
    - median_residual (median absolute error)
    - representative_sample_1 (25th percentile error)
    - representative_sample_2 (75th percentile error)

    Args:
        y_true: Ground truth target Series.
        y_pred: Predicted values array.
        max_samples: Maximum number of representative samples to return.

    Returns:
        List of tuples: (original_index: int, category_name: str).
    """
    y_t = np.array(y_true, dtype=float)
    residuals = np.abs(y_t - y_pred)
    sorted_order = np.argsort(residuals)

    n = len(residuals)
    if n == 0:
        return []

    low_idx = int(sorted_order[0])
    high_idx = int(sorted_order[-1])
    med_idx = int(sorted_order[n // 2])
    p25_idx = int(sorted_order[n // 4])
    p75_idx = int(sorted_order[(3 * n) // 4])

    candidates = [
        (low_idx, "low_residual"),
        (high_idx, "high_residual"),
        (med_idx, "median_residual"),
        (p25_idx, "representative_sample_1"),
        (p75_idx, "representative_sample_2"),
    ]

    selected: List[Tuple[int, str]] = []
    used_indices = set()

    for idx, cat_name in candidates:
        if idx not in used_indices:
            selected.append((idx, cat_name))
            used_indices.add(idx)
            if len(selected) >= max_samples:
                break

    # Fallback to remaining indices if needed
    if len(selected) < max_samples:
        for idx in range(n):
            if idx not in used_indices:
                selected.append((idx, "representative_sample"))
                used_indices.add(idx)
                if len(selected) >= max_samples:
                    break

    return selected[:max_samples]


def generate_local_explanations(
    pipeline_or_model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_trans: np.ndarray,
    feature_names: List[str],
    task_type: str,
    contribution_extractor_fn: Callable[[int], List[FeatureContribution]],
    base_value: Optional[float] = None,
    max_samples: int = 5
) -> List[LocalExplanationRecord]:
    """
    Generate local prediction explanation records for up to 5 representative samples.

    Args:
        pipeline_or_model: Fitted estimator or pipeline for making predictions.
        X_val: Feature DataFrame.
        y_val: Target Series.
        X_trans: Preprocessed feature array.
        feature_names: List of feature names.
        task_type: "classification" or "regression".
        contribution_extractor_fn: Function mapping original_row_index (int) to List[FeatureContribution].
        base_value: Optional base/expected value float.
        max_samples: Maximum number of local samples (default: 5).

    Returns:
        List of LocalExplanationRecord objects.
    """
    y_pred = pipeline_or_model.predict(X_val)
    y_pred_arr = np.array(y_pred)

    if task_type.lower() == "classification":
        samples = select_representative_samples_classification(y_val, y_pred_arr, max_samples=max_samples)
    else:
        samples = select_representative_samples_regression(y_val, y_pred_arr, max_samples=max_samples)

    local_records: List[LocalExplanationRecord] = []

    for orig_row_idx, cat_name in samples:
        # Access prediction using the exact original row index
        pred_val = y_pred_arr[orig_row_idx]
        actual_val = y_val.iloc[orig_row_idx] if orig_row_idx < len(y_val) else None

        # Clean scalar types for JSON safety
        if isinstance(pred_val, (np.generic, float)):
            pred_clean: Any = round(float(pred_val), 4) if isinstance(pred_val, float) else int(pred_val)
        else:
            pred_clean = pred_val

        if actual_val is not None:
            if isinstance(actual_val, float):
                actual_clean: Any = round(float(actual_val), 4)
            elif isinstance(actual_val, (int, np.integer)):
                actual_clean = int(actual_val)
            else:
                actual_clean = str(actual_val)
        else:
            actual_clean = None

        # Extract local contributions corresponding to orig_row_idx
        contributions = contribution_extractor_fn(orig_row_idx)

        local_records.append(
            LocalExplanationRecord(
                sample_index=orig_row_idx,
                category=cat_name,
                prediction=pred_clean,
                actual_value=actual_clean,
                base_value=round(base_value, 4) if base_value is not None else None,
                contributions=contributions
            )
        )

    return local_records
