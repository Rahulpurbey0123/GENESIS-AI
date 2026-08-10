"""
Permutation Feature Importance Module for GENESIS-AI Week 5 Explainability Engine.

Evaluates permutation feature importance on held-out validation data splits using scikit-learn's
permutation_importance inspector. Enforces strict research evaluation metrics:
- Classification: Macro F1 ('f1_macro')
- Regression: Negative RMSE ('neg_root_mean_squared_error')
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, mean_squared_error

from backend.explainability.schemas import FeatureImportanceRecord

logger = logging.getLogger("genesis.explainability.permutation")


def custom_f1_macro_scorer(estimator: Any, X: pd.DataFrame, y: Any) -> float:
    """
    Custom macro F1 scorer enforcing project research evaluation convention.
    Calculates macro-averaged F1 score across all target classes.
    """
    y_pred = estimator.predict(X)
    return float(f1_score(y, y_pred, average="macro", zero_division=0))


def custom_neg_rmse_scorer(estimator: Any, X: pd.DataFrame, y: Any) -> float:
    """
    Custom negative RMSE scorer for regression permutation importance.
    
    Note: Negative RMSE (-RMSE) is used internally because scikit-learn's permutation_importance
    assumes higher scores indicate better model performance, whereas lower RMSE indicates
    better predictive accuracy in research reporting.
    """
    y_pred = estimator.predict(X)
    y_num = np.array(y, dtype=float)
    y_p_num = np.array(y_pred, dtype=float)
    mse = float(mean_squared_error(y_num, y_p_num))
    return -float(np.sqrt(mse))


def explain_permutation_importance(
    pipeline_or_model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    task_type: str,
    n_repeats: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute global permutation feature importance on held-out validation set.

    Args:
        pipeline_or_model: Fitted Pipeline or model estimator with .predict method.
        X_val: Held-out validation feature DataFrame.
        y_val: Validation target Series.
        feature_names: List of feature names.
        task_type: "classification" or "regression".
        n_repeats: Number of permutation iterations per feature (default: 5).
        random_state: Seed for random permutations.

    Returns:
        Dict containing:
        - 'global_importance': List of FeatureImportanceRecord
        - 'mean_importance': Array of mean importances
        - 'std_importance': Array of std importances
        - 'scoring_metric': Exact research metric string ('f1_macro' or 'neg_root_mean_squared_error')
        - 'warnings': List of warnings
    """
    warnings: List[str] = []

    if task_type.lower() == "classification":
        scorer_fn = custom_f1_macro_scorer
        scoring_metric = "f1_macro"
    else:
        scorer_fn = custom_neg_rmse_scorer
        scoring_metric = "neg_root_mean_squared_error"

    try:
        res = permutation_importance(
            estimator=pipeline_or_model,
            X=X_val,
            y=y_val,
            scoring=scorer_fn,
            n_repeats=n_repeats,
            random_state=random_state
        )
    except Exception as e:
        raise RuntimeError(f"Permutation importance failed with metric '{scoring_metric}': {str(e)}")

    raw_means = np.array(res.importances_mean)
    raw_stds = np.array(res.importances_std)

    # Clean small negative importances due to random sampling noise for global ranking
    cleaned_means = np.maximum(0.0, raw_means)
    total_mean = np.sum(cleaned_means)

    if total_mean > 0:
        normalized_imp = cleaned_means / total_mean
    else:
        normalized_imp = np.zeros_like(cleaned_means)

    # Sort features by rank descending
    sorted_indices = np.argsort(normalized_imp)[::-1]
    global_importance_records: List[FeatureImportanceRecord] = []

    for rank_idx, col_i in enumerate(sorted_indices, start=1):
        fname = feature_names[col_i] if col_i < len(feature_names) else f"feature_{col_i}"
        imp_score = round(float(normalized_imp[col_i]), 4)
        m_val = round(float(raw_means[col_i]), 4)
        s_val = round(float(raw_stds[col_i]), 4)

        global_importance_records.append(
            FeatureImportanceRecord(
                feature=fname,
                importance=imp_score,
                rank=rank_idx,
                direction=None,  # Permutation importance is a global model-level magnitude
                mean_importance=m_val,
                std_importance=s_val
            )
        )

    return {
        "global_importance": global_importance_records,
        "mean_importance": raw_means,
        "std_importance": raw_stds,
        "scoring_metric": scoring_metric,
        "n_repeats": n_repeats,
        "warnings": warnings
    }
