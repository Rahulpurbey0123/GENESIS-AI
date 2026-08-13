"""
Scikit-Learn Pipeline Evaluator for Evolutionary Optimization Engine v1.0.

Builds, trains, and evaluates candidate machine learning pipelines on train/validation dataset splits.
"""

from typing import Dict, Any, Tuple, Optional, List
import logging
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score

from backend.optimization.chromosome import Chromosome

logger = logging.getLogger("genesis.optimization.evaluator")


def build_estimator(
    pipeline_id: str,
    hyperparameters: Dict[str, Any],
    random_state: int = 42,
    n_samples: Optional[int] = None
) -> Any:
    """Instantiate scikit-learn model estimator configured with chromosome hyperparameters."""
    hp = dict(hyperparameters)

    if "k_neighbors" in pipeline_id and n_samples is not None and n_samples > 0:
        if "n_neighbors" in hp:
            hp["n_neighbors"] = max(1, min(hp["n_neighbors"], n_samples))

    # Classification
    if pipeline_id == "classification_logistic_regression":
        return LogisticRegression(**hp, random_state=random_state)
    elif pipeline_id == "classification_random_forest":
        return RandomForestClassifier(**hp, random_state=random_state)
    elif pipeline_id == "classification_hist_gradient_boosting":
        return HistGradientBoostingClassifier(**hp, random_state=random_state)
    elif pipeline_id == "classification_svc":
        return SVC(**hp, random_state=random_state)
    elif pipeline_id == "classification_k_neighbors":
        return KNeighborsClassifier(**hp)

    # Regression
    elif pipeline_id == "regression_linear_regression":
        return LinearRegression(**hp)
    elif pipeline_id == "regression_random_forest":
        return RandomForestRegressor(**hp, random_state=random_state)
    elif pipeline_id == "regression_hist_gradient_boosting":
        return HistGradientBoostingRegressor(**hp, random_state=random_state)
    elif pipeline_id == "regression_svr":
        return SVR(**hp)
    elif pipeline_id == "regression_k_neighbors":
        return KNeighborsRegressor(**hp)
    else:
        raise ValueError(f"Unknown pipeline_id '{pipeline_id}'")


def build_sklearn_pipeline(
    pipeline_id: str,
    hyperparameters: Dict[str, Any],
    X_sample: pd.DataFrame,
    random_state: int = 42
) -> Pipeline:
    """
    Construct a complete, runnable scikit-learn Pipeline with preprocessing and model estimator.

    Args:
        pipeline_id: Registered candidate pipeline identifier.
        hyperparameters: Validated hyperparameter configuration dict.
        X_sample: Input features DataFrame used to infer numeric vs categorical column types.
        random_state: Random seed.

    Returns:
        Configured scikit-learn Pipeline object.
    """
    estimator = build_estimator(
        pipeline_id=pipeline_id,
        hyperparameters=hyperparameters,
        random_state=random_state,
        n_samples=len(X_sample)
    )

    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(X_sample)

    num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_sample.select_dtypes(exclude=[np.number]).columns.tolist()

    requires_scaling = "svc" in pipeline_id or "svr" in pipeline_id or "logistic" in pipeline_id or "k_neighbors" in pipeline_id or "linear_regression" in pipeline_id
    use_ordinal = "hist_gradient_boosting" in pipeline_id

    transformers = []

    if num_cols:
        num_steps = [
            ("imputer_median", SimpleImputer(strategy="median")),
            ("imputer_fallback", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]
        if requires_scaling:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))

    if cat_cols:
        if use_ordinal:
            cat_steps = [
                ("imputer_freq", SimpleImputer(strategy="most_frequent")),
                ("imputer_fallback", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        else:
            cat_steps = [
                ("imputer_freq", SimpleImputer(strategy="most_frequent")),
                ("imputer_fallback", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))


    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
    else:
        full_pipeline = Pipeline([("model", estimator)])

    return full_pipeline


def evaluate_chromosome(
    chromosome: Chromosome,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_type: str,
    random_state: int = 42,
    metric_name: Optional[str] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Fit candidate pipeline on training split and evaluate fitness score on validation split.

    Args:
        chromosome: Chromosome object to evaluate.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_val: Validation feature DataFrame.
        y_val: Validation target Series.
        task_type: "classification" or "regression".
        random_state: Seed for model estimators.
        metric_name: Optional custom metric override.

    Returns:
        Tuple of (fitness_score, detailed_metrics_dict).
    """
    try:
        pipeline = build_sklearn_pipeline(
            pipeline_id=chromosome.pipeline_id,
            hyperparameters=chromosome.hyperparameters,
            X_sample=X_train,
            random_state=random_state
        )

        # Fit model on training split
        pipeline.fit(X_train, y_train)

        # Predict on validation split
        y_pred = pipeline.predict(X_val)

        metrics: Dict[str, float] = {}

        if task_type.lower() == "classification":
            acc = float(accuracy_score(y_val, y_pred))
            metrics["accuracy"] = round(acc, 4)

            # Determine F1 score average strategy
            unique_classes = np.unique(y_train)
            if len(unique_classes) == 2:
                f1_val = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
            else:
                f1_val = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

            metrics["f1"] = round(f1_val, 4)
            fitness = round(f1_val, 4)

            if metric_name == "accuracy":
                fitness = round(acc, 4)

        else:  # regression
            mse = float(mean_squared_error(y_val, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_val, y_pred))
            try:
                r2 = float(r2_score(y_val, y_pred))
            except Exception:
                r2 = -1.0

            metrics["rmse"] = round(rmse, 4)
            metrics["mae"] = round(mae, 4)
            metrics["r2"] = round(r2, 4)

            # GA maximizes fitness, so fitness is -RMSE
            fitness = round(-rmse, 4)

        return fitness, metrics

    except Exception as e:
        logger.warning(f"Evaluation failed for chromosome {chromosome}: {str(e)}")
        return float("-inf"), {
            "status": "failed",
            "fitness": float("-inf"),
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
