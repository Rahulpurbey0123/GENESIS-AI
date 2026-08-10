"""
Discrete Hyperparameter Search Spaces for Evolutionary Optimization Engine v1.0.

Defines reproducible hyperparameter grids for all candidate machine learning pipelines.
"""

from typing import Dict, Any, List, Optional
import random
import numpy as np


SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    # Classification
    "classification_logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs"],
        "max_iter": [500, 1000, 2000],
    },
    "classification_random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "classification_hist_gradient_boosting": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 300],
        "max_leaf_nodes": [15, 31, 63],
        "max_depth": [None, 5, 10],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "classification_svc": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "classification_k_neighbors": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    # Regression
    "regression_linear_regression": {
        "fit_intercept": [True, False],
    },
    "regression_random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "regression_hist_gradient_boosting": {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 300],
        "max_leaf_nodes": [15, 31, 63],
        "max_depth": [None, 5, 10],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "regression_svr": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1, 0.2],
    },
    "regression_k_neighbors": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}


def get_search_space(pipeline_id: str) -> Dict[str, List[Any]]:
    """
    Retrieve discrete hyperparameter search space grid for a given pipeline_id.

    Args:
        pipeline_id: Registered pipeline identifier string.

    Returns:
        Dictionary mapping hyperparameter names to list of candidate values.
    """
    if pipeline_id not in SEARCH_SPACES:
        raise ValueError(f"Unknown pipeline_id '{pipeline_id}'. Not found in search space registry.")
    return SEARCH_SPACES[pipeline_id]


def sample_random_hyperparameters(
    pipeline_id: str,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Sample a random valid hyperparameter configuration for a pipeline from its discrete search space grid.

    Args:
        pipeline_id: Registered pipeline identifier.
        rng: Optional random.Random instance for deterministic sampling.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    grid = get_search_space(pipeline_id)
    if rng is None:
        rng = random.Random()

    params = {}
    for param_name, values in grid.items():
        params[param_name] = rng.choice(values)
    return params


def validate_hyperparameters(pipeline_id: str, hyperparameters: Dict[str, Any]) -> bool:
    """
    Validate whether a hyperparameter configuration dictionary is valid for the given pipeline_id.

    Args:
        pipeline_id: Registered pipeline identifier.
        hyperparameters: Dictionary of hyperparameter values to check.

    Returns:
        True if all parameters exist in grid and have valid values; False otherwise.
    """
    try:
        grid = get_search_space(pipeline_id)
    except ValueError:
        return False

    for param_name, param_val in hyperparameters.items():
        if param_name not in grid:
            return False
        if param_val not in grid[param_name]:
            return False

    return True
