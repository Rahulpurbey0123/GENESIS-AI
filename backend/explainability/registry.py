"""
Model Explanation Strategy Registry for GENESIS-AI Week 5 Explainability Engine.

Maps scikit-learn model estimator classes to appropriate explanation strategies:
- 'shap_tree': SHAP TreeExplainer for tree ensembles.
- 'linear_coefficients': Coefficient extraction for linear models.
- 'native_tree': Native feature_importances_ extraction for tree models.
- 'permutation_importance': Permutation importance fallback on validation split.
"""

from typing import Dict, Any, Optional, Type
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# Registered model explanation mapping
MODEL_STRATEGY_MAP: Dict[str, str] = {
    # Linear models -> linear_coefficients
    "LogisticRegression": "linear_coefficients",
    "LinearRegression": "linear_coefficients",

    # Tree ensemble models -> shap_tree
    "RandomForestClassifier": "shap_tree",
    "RandomForestRegressor": "shap_tree",
    "HistGradientBoostingClassifier": "shap_tree",
    "HistGradientBoostingRegressor": "shap_tree",

    # Non-linear non-tree models -> permutation_importance
    "SVC": "permutation_importance",
    "SVR": "permutation_importance",
    "KNeighborsClassifier": "permutation_importance",
    "KNeighborsRegressor": "permutation_importance",
}


class ExplanationRegistry:
    """Registry mapping estimator classes and names to explanation strategies."""

    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None):
        self._mapping = dict(custom_mapping or MODEL_STRATEGY_MAP)

    def get_strategy(self, model: Any) -> str:
        """
        Determine the primary explanation strategy for a fitted estimator instance or class.

        Args:
            model: Scikit-learn estimator instance or class.

        Returns:
            Strategy string name. Defaults to 'permutation_importance' for unknown models if fitted,
            or 'unsupported' if invalid.
        """
        if model is None:
            return "unsupported"

        class_name = model.__class__.__name__ if not isinstance(model, type) else model.__name__

        if class_name in self._mapping:
            return self._mapping[class_name]

        # Check by inheritance / Duck typing
        if hasattr(model, "coef_"):
            return "linear_coefficients"
        if hasattr(model, "feature_importances_"):
            return "native_tree"

        # Fallback for unknown tabular estimators
        if hasattr(model, "predict"):
            return "permutation_importance"

        return "unsupported"

    def register_strategy(self, model_name: str, strategy: str) -> None:
        """Register or override explanation strategy for a model class name."""
        valid_strategies = {"shap_tree", "linear_coefficients", "native_tree", "permutation_importance", "unsupported"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
        self._mapping[model_name] = strategy

    def is_supported(self, model: Any) -> bool:
        """Check if model is supported by any explanation strategy."""
        return self.get_strategy(model) != "unsupported"
