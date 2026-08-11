"""
Evaluation Metrics Module for GENESIS-AI Week 7 Research Evaluation.

Provides consistent metric computation for classification and regression tasks,
efficiency metrics (candidate count, search space reduction, runtime), and metric comparison utilities.
"""

from typing import Dict, Any, Union, Tuple, List
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any
) -> Dict[str, float]:
    """
    Calculate predictive performance metrics for classification tasks.

    Returns:
        Dict containing 'f1' (Macro F1) and 'accuracy'.
    """
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    return {
        "f1": round(f1, 4),
        "accuracy": round(acc, 4)
    }


def calculate_regression_metrics(
    y_true: Any,
    y_pred: Any
) -> Dict[str, float]:
    """
    Calculate predictive performance metrics for regression tasks.

    Returns:
        Dict containing 'rmse', 'mae', and 'r2'.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }


def get_primary_metric_name(task_type: str) -> str:
    """Return primary evaluation metric name based on task type."""
    if task_type.lower() == "classification":
        return "f1"
    return "rmse"


def get_metric_direction(metric_name: str) -> str:
    """
    Return optimization direction ('higher' or 'lower') for a given metric.

    Raises:
        ValueError: If metric_name is unrecognized.
    """
    name = metric_name.lower().strip()
    if name in ["macro_f1", "f1", "accuracy", "r2"]:
        return "higher"
    elif name in ["rmse", "mae", "mse", "neg_root_mean_squared_error"]:
        return "lower"
    raise ValueError(f"Unknown metric '{metric_name}'. Metric direction must be explicitly configured.")


def is_higher_better(metric_name: str) -> bool:
    """Determine if a higher metric score indicates superior performance."""
    return get_metric_direction(metric_name) == "higher"


def is_better_score(score_a: float, score_b: float, metric_name: str) -> bool:
    """Compare two scores according to the metric's directionality."""
    direction = get_metric_direction(metric_name)
    if direction == "higher":
        return score_a > score_b
    return score_a < score_b


def calculate_efficiency_metrics(
    candidate_count_before: int,
    candidate_count_after: int,
    evaluations_used: int,
    runtime_seconds: float
) -> Dict[str, Any]:
    """
    Calculate search space reduction and efficiency indicators.
    """
    if candidate_count_before > 0:
        reduction = round((candidate_count_before - candidate_count_after) / candidate_count_before, 4)
    else:
        reduction = 0.0

    return {
        "candidate_count_before": candidate_count_before,
        "candidate_count_after": candidate_count_after,
        "candidate_space_reduction": reduction,
        "evaluations_used": evaluations_used,
        "runtime_seconds": round(runtime_seconds, 2)
    }
