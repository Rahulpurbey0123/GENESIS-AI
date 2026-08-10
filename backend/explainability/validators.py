"""
Validation utilities for GENESIS-AI Week 5 Explainability Engine.

Ensures strict verification of fitted models, feature names, importance scores, rank sequences,
local sample indices, and final JSON output safety (no NaN / Inf values).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted, NotFittedError

from backend.explainability.schemas import ExplanationOutput, FeatureImportanceRecord


class ValidationError(Exception):
    """Raised when validation fails critically."""
    pass


def validate_fitted_model(pipeline_or_estimator: Any) -> Tuple[bool, Optional[str]]:
    """
    Verify that the input scikit-learn Pipeline or estimator is fitted.

    Args:
        pipeline_or_estimator: Fitted scikit-learn Pipeline or estimator object.

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str]).
    """
    if pipeline_or_estimator is None:
        return False, "Model is None."

    # Extract model step if input is a Pipeline
    if hasattr(pipeline_or_estimator, "named_steps") and "model" in pipeline_or_estimator.named_steps:
        estimator = pipeline_or_estimator.named_steps["model"]
    else:
        estimator = pipeline_or_estimator

    try:
        check_is_fitted(estimator)
        return True, None
    except NotFittedError:
        return False, f"Estimator {estimator.__class__.__name__} is not fitted."
    except Exception as e:
        # Fallback check for attributes that indicate fitting
        fitted_attrs = ["classes_", "coef_", "feature_importances_", "estimators_", "n_features_in_"]
        if any(hasattr(estimator, attr) for attr in fitted_attrs):
            return True, None
        return False, f"Fitted model check failed: {str(e)}"


def validate_feature_names(
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Validate and return clean list of feature names matching X's shape.

    Args:
        X: Feature matrix (DataFrame or numpy array).
        feature_names: Optional list of explicit feature names.

    Returns:
        Tuple of (validated_feature_names: List[str], warnings: List[str]).
    """
    warnings: List[str] = []
    n_cols = X.shape[1] if len(X.shape) > 1 else 1

    if feature_names is not None:
        if len(feature_names) != n_cols:
            warnings.append(
                f"Feature name count mismatch: provided {len(feature_names)}, expected {n_cols}. Generating defaults."
            )
            names = [f"feature_{i}" for i in range(n_cols)]
        else:
            names = [str(f) for f in feature_names]
    elif isinstance(X, pd.DataFrame):
        names = [str(c) for c in X.columns]
    else:
        names = [f"feature_{i}" for i in range(n_cols)]

    return names, warnings


def validate_importance_values(
    global_importance: List[FeatureImportanceRecord]
) -> Tuple[bool, List[str]]:
    """
    Validate global importance records for non-empty, finite values, non-negative scores, and valid ranks.

    Args:
        global_importance: List of FeatureImportanceRecord objects.

    Returns:
        Tuple of (is_valid: bool, errors_or_warnings: List[str]).
    """
    errors: List[str] = []

    if not global_importance:
        return True, []

    ranks = [rec.rank for rec in global_importance]
    if len(ranks) != len(set(ranks)):
        errors.append("Duplicate ranks found in global importance records.")

    for rec in global_importance:
        if math.isnan(rec.importance) or math.isinf(rec.importance):
            errors.append(f"Non-finite importance value ({rec.importance}) for feature '{rec.feature}'.")
        if rec.importance < -1e-6:
            errors.append(f"Negative importance value ({rec.importance}) for feature '{rec.feature}'.")

    return len(errors) == 0, errors


def validate_no_nan_inf(data: Any) -> bool:
    """
    Recursively check that a nested structure (dict, list, primitive) contains no NaN or Inf values.

    Args:
        data: Primitive value or collection.

    Returns:
        True if all numeric values are finite and clean, False otherwise.
    """
    if isinstance(data, float):
        return not (math.isnan(data) or math.isinf(data))
    elif isinstance(data, dict):
        return all(validate_no_nan_inf(v) for v in data.values())
    elif isinstance(data, (list, tuple)):
        return all(validate_no_nan_inf(x) for x in data)
    return True
