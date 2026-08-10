"""
SHAP TreeExplainer Module for GENESIS-AI Week 5 Explainability Engine.

Isolates SHAP TreeExplainer logic for tree-based ensemble models (RandomForest, HistGradientBoosting).
Computes global SHAP feature importances, local feature contributions, and expected base values.
Enforces strict original-row alignment without silent row substitutions.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from backend.explainability.schemas import FeatureImportanceRecord, FeatureContribution

logger = logging.getLogger("genesis.explainability.shap")


def explain_shap_tree(
    model: Any,
    X_trans: np.ndarray,
    feature_names: List[str],
    raw_sample_df: Optional[pd.DataFrame] = None,
    max_samples: int = 200,
    eval_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Compute SHAP TreeExplainer global feature importances and local contributions.

    Args:
        model: Fitted tree-based scikit-learn estimator.
        X_trans: Preprocessed feature array matching the model input shape.
        feature_names: List of feature names corresponding to X_trans columns.
        raw_sample_df: Optional raw feature DataFrame for displaying human-readable feature values.
        max_samples: Maximum number of rows to evaluate with SHAP for performance.
        eval_indices: Optional list of specific original row indices to evaluate (preserves local alignment).

    Returns:
        Dict containing:
        - 'global_importance': List of FeatureImportanceRecord
        - 'vals_array': Raw SHAP values array of shape (n_eval_samples, n_features)
        - 'base_value': Base/expected value float
        - 'index_to_eval_pos': Dict mapping orig_row_idx -> pos in vals_array
        - 'warnings': List of warning strings
    """
    if not HAS_SHAP:
        raise RuntimeError("SHAP library is not installed.")

    warnings: List[str] = []

    # Determine subset of rows to evaluate
    if eval_indices is not None:
        effective_indices = list(dict.fromkeys(eval_indices))[:max_samples]
    else:
        if len(X_trans) > max_samples:
            effective_indices = list(range(max_samples))
        else:
            effective_indices = list(range(len(X_trans)))

    index_to_eval_pos = {orig_idx: pos for pos, orig_idx in enumerate(effective_indices)}
    X_eval = X_trans[effective_indices]

    # Initialize SHAP TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_res = explainer(X_eval)
    except Exception as e:
        # Fallback to direct shap_values call if explainer call signature differs
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_eval)
            base_value = getattr(explainer, "expected_value", 0.0)
            shap_res = None
        except Exception as err:
            raise RuntimeError(f"SHAP TreeExplainer failed for model {model.__class__.__name__}: {str(err)}")

    # Standardize SHAP output format into array and base_value
    if shap_res is not None:
        raw_vals = shap_res.values
        base_vals = shap_res.base_values
    else:
        raw_vals = shap_values
        base_vals = base_value

    # Process raw SHAP values shape:
    # 1. Binary / Multiclass classification with list of arrays [class0, class1, ...]
    if isinstance(raw_vals, list):
        if len(raw_vals) == 2:
            # Binary classification: focus on positive class (class 1)
            vals_array = raw_vals[1]
        else:
            # Multiclass: mean absolute SHAP value across all classes
            vals_array = np.mean([np.abs(v) for v in raw_vals], axis=0)
    elif isinstance(raw_vals, np.ndarray):
        if len(raw_vals.shape) == 3:
            # (n_samples, n_features, n_classes)
            if raw_vals.shape[2] == 2:
                vals_array = raw_vals[:, :, 1]
            else:
                vals_array = np.mean(np.abs(raw_vals), axis=2)
        else:
            # (n_samples, n_features)
            vals_array = raw_vals
    else:
        vals_array = np.array(raw_vals)

    # Standardize base value float
    if isinstance(base_vals, (list, np.ndarray)):
        b_arr = np.array(base_vals).flatten()
        base_val_float = float(b_arr[1]) if len(b_arr) > 1 else float(b_arr[0])
    else:
        try:
            base_val_float = float(base_vals)
        except Exception:
            base_val_float = 0.0

    # Compute global mean absolute SHAP feature importances
    mean_abs_shap = np.mean(np.abs(vals_array), axis=0)
    total_shap = np.sum(mean_abs_shap)

    if total_shap > 0:
        normalized_imp = mean_abs_shap / total_shap
    else:
        normalized_imp = np.zeros_like(mean_abs_shap)

    # Build global importance records sorted by rank
    sorted_indices = np.argsort(normalized_imp)[::-1]
    global_importance_records: List[FeatureImportanceRecord] = []

    for rank_idx, col_i in enumerate(sorted_indices, start=1):
        fname = feature_names[col_i] if col_i < len(feature_names) else f"feature_{col_i}"
        imp_score = round(float(normalized_imp[col_i]), 4)
        global_importance_records.append(
            FeatureImportanceRecord(
                feature=fname,
                importance=imp_score,
                rank=rank_idx,
                direction=None  # SHAP global importance magnitude is non-directional
            )
        )

    return {
        "global_importance": global_importance_records,
        "vals_array": vals_array,
        "base_value": round(base_val_float, 4),
        "index_to_eval_pos": index_to_eval_pos,
        "eval_count": len(X_eval),
        "warnings": warnings
    }


def get_local_shap_contributions(
    vals_array: np.ndarray,
    orig_row_idx: int,
    feature_names: List[str],
    X_trans: np.ndarray,
    raw_sample_df: Optional[pd.DataFrame] = None,
    index_to_eval_pos: Optional[Dict[int, int]] = None
) -> List[FeatureContribution]:
    """
    Extract ranked feature contributions for a single sample prediction using SHAP values.
    Enforces strict sample index alignment without silent row substitutions.

    Args:
        vals_array: Evaluated SHAP values array of shape (n_eval_samples, n_features).
        orig_row_idx: Exact original row index in dataset.
        feature_names: List of feature names matching X_trans.
        X_trans: Preprocessed feature array matching model input.
        raw_sample_df: Optional raw feature DataFrame.
        index_to_eval_pos: Dict mapping orig_row_idx -> pos in vals_array.

    Returns:
        List of FeatureContribution sorted by absolute contribution magnitude descending.
    """
    if index_to_eval_pos is not None:
        if orig_row_idx not in index_to_eval_pos:
            raise KeyError(f"SHAP explanation was not evaluated for original row index {orig_row_idx}")
        row_pos = index_to_eval_pos[orig_row_idx]
    else:
        if orig_row_idx < len(vals_array):
            row_pos = orig_row_idx
        else:
            raise KeyError(f"SHAP explanation was not evaluated for original row index {orig_row_idx}")

    sample_shap = vals_array[row_pos]
    contributions: List[FeatureContribution] = []

    for col_i, shap_val in enumerate(sample_shap):
        fname = feature_names[col_i] if col_i < len(feature_names) else f"feature_{col_i}"
        
        # Extract feature value if available from raw DataFrame or transformed array
        feat_val = None
        if raw_sample_df is not None and fname in raw_sample_df.columns and orig_row_idx < len(raw_sample_df):
            raw_v = raw_sample_df.iloc[orig_row_idx][fname]
            if isinstance(raw_v, (int, float, str, bool, np.generic)):
                feat_val = raw_v if not pd.isna(raw_v) else None
        elif orig_row_idx < X_trans.shape[0] and col_i < X_trans.shape[1]:
            val_num = X_trans[orig_row_idx, col_i]
            feat_val = round(float(val_num), 4) if not np.isnan(val_num) else None

        contributions.append(
            FeatureContribution(
                feature=fname,
                feature_value=feat_val,
                contribution=round(float(shap_val), 4)
            )
        )

    # Sort contributions by absolute magnitude descending
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    return contributions
