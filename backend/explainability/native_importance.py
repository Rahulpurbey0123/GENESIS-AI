"""
Native Feature Importance & Linear Coefficient Module for GENESIS-AI Week 5 Explainability Engine.

Extracts feature_importances_ from tree models and coef_ weights from linear models.
Computes normalized global feature importances, feature direction indicators, and local linear sample contributions.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from backend.explainability.schemas import FeatureImportanceRecord, FeatureContribution


def explain_linear_coefficients(
    model: Any,
    X_trans: np.ndarray,
    feature_names: List[str],
    raw_sample_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Extract linear regression / logistic regression model coefficients.

    Args:
        model: Fitted linear model exposing coef_ attribute.
        X_trans: Preprocessed feature array matching model input.
        feature_names: List of feature names matching X_trans columns.
        raw_sample_df: Optional raw feature DataFrame.

    Returns:
        Dict containing 'global_importance', 'raw_weights', 'intercept', and 'warnings'.
    """
    warnings: List[str] = []

    if not hasattr(model, "coef_"):
        raise ValueError(f"Model {model.__class__.__name__} does not expose coef_.")

    coef = np.array(model.coef_)

    # Intercept
    intercept = getattr(model, "intercept_", 0.0)
    if isinstance(intercept, np.ndarray):
        intercept_val = float(intercept[0]) if len(intercept) > 0 else 0.0
    else:
        intercept_val = float(intercept)

    # Multi-class vs Binary vs Regression
    if len(coef.shape) == 2:
        if coef.shape[0] == 1:
            # Binary classification or 1D target: shape (1, n_features)
            raw_weights = coef[0]
            is_multiclass = False
        else:
            # Multi-class: shape (n_classes, n_features) -> mean absolute weight across classes
            raw_weights = np.mean(np.abs(coef), axis=0)
            is_multiclass = True
    else:
        raw_weights = coef
        is_multiclass = False

    abs_weights = np.abs(raw_weights)
    sum_abs = np.sum(abs_weights)

    if sum_abs > 0:
        normalized_imp = abs_weights / sum_abs
    else:
        normalized_imp = np.zeros_like(abs_weights)

    # Sort features by importance rank descending
    sorted_indices = np.argsort(normalized_imp)[::-1]
    global_importance_records: List[FeatureImportanceRecord] = []

    for rank_idx, col_i in enumerate(sorted_indices, start=1):
        fname = feature_names[col_i] if col_i < len(feature_names) else f"feature_{col_i}"
        imp_score = round(float(normalized_imp[col_i]), 4)

        if not is_multiclass:
            w_val = float(raw_weights[col_i])
            direction = 1 if w_val > 0 else (-1 if w_val < 0 else 0)
        else:
            direction = None  # Multi-class aggregate magnitude has no single direction

        global_importance_records.append(
            FeatureImportanceRecord(
                feature=fname,
                importance=imp_score,
                rank=rank_idx,
                direction=direction
            )
        )

    return {
        "global_importance": global_importance_records,
        "raw_weights": raw_weights,
        "intercept": round(intercept_val, 4),
        "is_multiclass": is_multiclass,
        "warnings": warnings
    }


def get_local_linear_contributions(
    raw_weights: np.ndarray,
    sample_row_trans: np.ndarray,
    feature_names: List[str],
    raw_sample_df: Optional[pd.DataFrame] = None,
    orig_row_idx: int = 0
) -> List[FeatureContribution]:
    """
    Compute local linear feature contributions for a single sample prediction: contribution_i = weight_i * x_trans_i.

    Args:
        raw_weights: 1D array of feature weights in transformed space.
        sample_row_trans: 1D array of transformed feature values for this exact sample.
        feature_names: List of feature names matching transformed features.
        raw_sample_df: Optional raw feature DataFrame.
        orig_row_idx: Original DataFrame row index.

    Returns:
        List of FeatureContribution sorted by absolute contribution magnitude descending.
    """
    contributions: List[FeatureContribution] = []

    for col_i, weight in enumerate(raw_weights):
        fname = feature_names[col_i] if col_i < len(feature_names) else f"feature_{col_i}"
        x_val = sample_row_trans[col_i] if col_i < len(sample_row_trans) else 0.0

        contrib_val = float(weight * x_val)

        feat_val = None
        if raw_sample_df is not None and fname in raw_sample_df.columns and orig_row_idx < len(raw_sample_df):
            raw_v = raw_sample_df.iloc[orig_row_idx][fname]
            if isinstance(raw_v, (int, float, str, bool, np.generic)):
                feat_val = raw_v if not pd.isna(raw_v) else None
        else:
            feat_val = round(float(x_val), 4) if not np.isnan(x_val) else None

        contributions.append(
            FeatureContribution(
                feature=fname,
                feature_value=feat_val,
                contribution=round(contrib_val, 4)
            )
        )

    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    return contributions


def explain_native_tree_importance(
    model: Any,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Extract native tree feature_importances_ (Gini / MDI importance).

    Args:
        model: Fitted tree model exposing feature_importances_.
        feature_names: List of feature names.

    Returns:
        Dict containing 'global_importance' records and 'warnings'.
    """
    warnings: List[str] = []

    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model {model.__class__.__name__} does not expose feature_importances_.")

    raw_imp = np.array(model.feature_importances_)
    sum_imp = np.sum(raw_imp)

    if sum_imp > 0:
        normalized_imp = raw_imp / sum_imp
    else:
        normalized_imp = np.zeros_like(raw_imp)

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
                direction=None  # Gini tree importance has no direction
            )
        )

    return {
        "global_importance": global_importance_records,
        "warnings": warnings
    }
