"""
GENESIS-AI Week 5 Explainability & Model Insight Engine Package.

Provides post-hoc explanations for fitted machine learning pipelines using SHAP TreeExplainer,
linear coefficients, native feature importance, and permutation importance.
"""

from backend.explainability.schemas import (
    ExplanationOutput,
    FeatureImportanceRecord,
    LocalExplanationRecord,
    FeatureContribution,
)
from backend.explainability.registry import ExplanationRegistry, MODEL_STRATEGY_MAP
from backend.explainability.validators import (
    validate_fitted_model,
    validate_feature_names,
    validate_importance_values,
    validate_no_nan_inf,
    ValidationError,
)
from backend.explainability.engine import ExplainabilityEngine

__all__ = [
    "ExplainabilityEngine",
    "ExplanationRegistry",
    "MODEL_STRATEGY_MAP",
    "ExplanationOutput",
    "FeatureImportanceRecord",
    "LocalExplanationRecord",
    "FeatureContribution",
    "validate_fitted_model",
    "validate_feature_names",
    "validate_importance_values",
    "validate_no_nan_inf",
    "ValidationError",
]
