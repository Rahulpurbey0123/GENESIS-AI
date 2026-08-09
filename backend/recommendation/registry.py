"""
Candidate Model and Pipeline Registry for GENESIS-AI Recommendation Engine v1.0.

Provides deterministic definitions and structured metadata for candidate machine learning
pipelines using scikit-learn base estimators (Classification and Regression).
"""

from typing import Dict, List, Optional
from backend.recommendation.schemas import PipelineMetadata, PipelineStep


CLASSIFICATION_PIPELINES: List[PipelineMetadata] = [
    PipelineMetadata(
        pipeline_id="classification_logistic_regression",
        name="Logistic Regression Pipeline",
        task="classification",
        model_family="linear",
        model_name="LogisticRegression",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="scaler",
                step_type="scaler",
                name="Standard Scaler",
                class_name="sklearn.preprocessing.StandardScaler"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="Logistic Regression Classifier",
                class_name="sklearn.linear_model.LogisticRegression"
            ),
        ],
        requires_scaling=True,
        supports_class_weight=True,
        handles_categorical_natively=False,
        nonlinear=False,
        computational_cost="low",
        small_data_suitable=True,
        large_data_suitable=True,
        high_dimensional_suitable=True,
    ),
    PipelineMetadata(
        pipeline_id="classification_random_forest",
        name="Random Forest Classifier Pipeline",
        task="classification",
        model_family="tree_ensemble",
        model_name="RandomForestClassifier",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="One-Hot Encoder",
                class_name="sklearn.preprocessing.OneHotEncoder"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="Random Forest Classifier",
                class_name="sklearn.ensemble.RandomForestClassifier"
            ),
        ],
        requires_scaling=False,
        supports_class_weight=True,
        handles_categorical_natively=False,
        nonlinear=True,
        computational_cost="medium",
        small_data_suitable=True,
        large_data_suitable=True,
        high_dimensional_suitable=False,
    ),
    PipelineMetadata(
        pipeline_id="classification_hist_gradient_boosting",
        name="HistGradientBoosting Classifier Pipeline",
        task="classification",
        model_family="tree_ensemble",
        model_name="HistGradientBoostingClassifier",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="Ordinal Encoder",
                class_name="sklearn.preprocessing.OrdinalEncoder"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="HistGradientBoosting Classifier",
                class_name="sklearn.ensemble.HistGradientBoostingClassifier"
            ),
        ],
        requires_scaling=False,
        supports_class_weight=True,
        handles_categorical_natively=False,
        nonlinear=True,
        computational_cost="low",
        small_data_suitable=True,
        large_data_suitable=True,
        high_dimensional_suitable=False,
    ),
    PipelineMetadata(
        pipeline_id="classification_svc",
        name="Support Vector Classifier Pipeline",
        task="classification",
        model_family="svm",
        model_name="SVC",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="One-Hot Encoder",
                class_name="sklearn.preprocessing.OneHotEncoder"
            ),
            PipelineStep(
                step_id="scaler",
                step_type="scaler",
                name="Standard Scaler",
                class_name="sklearn.preprocessing.StandardScaler"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="Support Vector Classifier",
                class_name="sklearn.svm.SVC"
            ),
        ],
        requires_scaling=True,
        supports_class_weight=True,
        handles_categorical_natively=False,
        nonlinear=True,
        computational_cost="high",
        small_data_suitable=True,
        large_data_suitable=False,
        high_dimensional_suitable=True,
    ),
    PipelineMetadata(
        pipeline_id="classification_k_neighbors",
        name="K-Neighbors Classifier Pipeline",
        task="classification",
        model_family="knn",
        model_name="KNeighborsClassifier",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="One-Hot Encoder",
                class_name="sklearn.preprocessing.OneHotEncoder"
            ),
            PipelineStep(
                step_id="scaler",
                step_type="scaler",
                name="Standard Scaler",
                class_name="sklearn.preprocessing.StandardScaler"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="K-Neighbors Classifier",
                class_name="sklearn.neighbors.KNeighborsClassifier"
            ),
        ],
        requires_scaling=True,
        supports_class_weight=False,
        handles_categorical_natively=False,
        nonlinear=True,
        computational_cost="medium",
        small_data_suitable=True,
        large_data_suitable=False,
        high_dimensional_suitable=False,
    ),
]


REGRESSION_PIPELINES: List[PipelineMetadata] = [
    PipelineMetadata(
        pipeline_id="regression_linear_regression",
        name="Linear Regression Pipeline",
        task="regression",
        model_family="linear",
        model_name="LinearRegression",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="One-Hot Encoder",
                class_name="sklearn.preprocessing.OneHotEncoder"
            ),
            PipelineStep(
                step_id="scaler",
                step_type="scaler",
                name="Standard Scaler",
                class_name="sklearn.preprocessing.StandardScaler"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="Linear Regression",
                class_name="sklearn.linear_model.LinearRegression"
            ),
        ],
        requires_scaling=True,
        supports_class_weight=False,
        handles_categorical_natively=False,
        nonlinear=False,
        computational_cost="low",
        small_data_suitable=True,
        large_data_suitable=True,
        high_dimensional_suitable=True,
    ),
    PipelineMetadata(
        pipeline_id="regression_random_forest",
        name="Random Forest Regressor Pipeline",
        task="regression",
        model_family="tree_ensemble",
        model_name="RandomForestRegressor",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="One-Hot Encoder",
                class_name="sklearn.preprocessing.OneHotEncoder"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="Random Forest Regressor",
                class_name="sklearn.ensemble.RandomForestRegressor"
            ),
        ],
        requires_scaling=False,
        supports_class_weight=False,
        handles_categorical_natively=False,
        nonlinear=True,
        computational_cost="medium",
        small_data_suitable=True,
        large_data_suitable=True,
        high_dimensional_suitable=False,
    ),
    PipelineMetadata(
        pipeline_id="regression_hist_gradient_boosting",
        name="HistGradientBoosting Regressor Pipeline",
        task="regression",
        model_family="tree_ensemble",
        model_name="HistGradientBoostingRegressor",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="Ordinal Encoder",
                class_name="sklearn.preprocessing.OrdinalEncoder"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="HistGradientBoosting Regressor",
                class_name="sklearn.ensemble.HistGradientBoostingRegressor"
            ),
        ],
        requires_scaling=False,
        supports_class_weight=False,
        handles_categorical_natively=False,
        nonlinear=True,
        computational_cost="low",
        small_data_suitable=True,
        large_data_suitable=True,
        high_dimensional_suitable=False,
    ),
    PipelineMetadata(
        pipeline_id="regression_svr",
        name="Support Vector Regressor Pipeline",
        task="regression",
        model_family="svm",
        model_name="SVR",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="One-Hot Encoder",
                class_name="sklearn.preprocessing.OneHotEncoder"
            ),
            PipelineStep(
                step_id="scaler",
                step_type="scaler",
                name="Standard Scaler",
                class_name="sklearn.preprocessing.StandardScaler"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="Support Vector Regressor",
                class_name="sklearn.svm.SVR"
            ),
        ],
        requires_scaling=True,
        supports_class_weight=False,
        handles_categorical_natively=False,
        nonlinear=True,
        computational_cost="high",
        small_data_suitable=True,
        large_data_suitable=False,
        high_dimensional_suitable=True,
    ),
    PipelineMetadata(
        pipeline_id="regression_k_neighbors",
        name="K-Neighbors Regressor Pipeline",
        task="regression",
        model_family="knn",
        model_name="KNeighborsRegressor",
        steps=[
            PipelineStep(
                step_id="imputer",
                step_type="imputer",
                name="Simple Imputer",
                class_name="sklearn.impute.SimpleImputer"
            ),
            PipelineStep(
                step_id="encoder",
                step_type="encoder",
                name="One-Hot Encoder",
                class_name="sklearn.preprocessing.OneHotEncoder"
            ),
            PipelineStep(
                step_id="scaler",
                step_type="scaler",
                name="Standard Scaler",
                class_name="sklearn.preprocessing.StandardScaler"
            ),
            PipelineStep(
                step_id="model",
                step_type="model",
                name="K-Neighbors Regressor",
                class_name="sklearn.neighbors.KNeighborsRegressor"
            ),
        ],
        requires_scaling=True,
        supports_class_weight=False,
        handles_categorical_natively=False,
        nonlinear=True,
        computational_cost="medium",
        small_data_suitable=True,
        large_data_suitable=False,
        high_dimensional_suitable=False,
    ),
]


class PipelineRegistry:
    """Central registry of candidate machine learning pipelines."""

    def __init__(self, custom_pipelines: Optional[List[PipelineMetadata]] = None):
        if custom_pipelines is not None:
            self._pipelines = {p.pipeline_id: p for p in custom_pipelines}
        else:
            self._pipelines = {
                p.pipeline_id: p
                for p in CLASSIFICATION_PIPELINES + REGRESSION_PIPELINES
            }

    def get_all_pipelines(self) -> List[PipelineMetadata]:
        """Return list of all registered pipeline candidate metadata."""
        return list(self._pipelines.values())

    def get_pipelines_by_task(self, task: str) -> List[PipelineMetadata]:
        """Return candidate pipelines matching the given task ('classification' or 'regression')."""
        task_lower = task.lower()
        return [p for p in self._pipelines.values() if p.task == task_lower]

    def get_pipeline_by_id(self, pipeline_id: str) -> Optional[PipelineMetadata]:
        """Fetch a specific pipeline metadata by pipeline_id."""
        return self._pipelines.get(pipeline_id)

    def total_count(self) -> int:
        """Return total number of registered candidate pipelines."""
        return len(self._pipelines)
