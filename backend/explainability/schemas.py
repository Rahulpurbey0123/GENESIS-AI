"""
Pydantic Schemas for GENESIS-AI Week 5 Explainability Engine.

Defines structured data contracts for global feature importances, local prediction explanations,
feature contributions, and complete explanation outputs.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class FeatureImportanceRecord(BaseModel):
    """Structured record for global feature importance."""
    feature: str = Field(description="Feature column name.")
    importance: float = Field(description="Normalized non-negative global feature importance score.")
    rank: int = Field(description="1-indexed rank order (1 is most important).")
    direction: Optional[int] = Field(
        default=None,
        description="Direction of impact: +1 for positive, -1 for negative, or None if unavailable/nonlinear."
    )
    mean_importance: Optional[float] = Field(
        default=None,
        description="Mean importance across permutation repeats (if permutation method used)."
    )
    std_importance: Optional[float] = Field(
        default=None,
        description="Standard deviation across permutation repeats (if permutation method used)."
    )


class FeatureContribution(BaseModel):
    """Structured record for a single feature's local contribution to a prediction."""
    feature: str = Field(description="Feature column name.")
    feature_value: Optional[Union[float, int, str, bool]] = Field(
        default=None,
        description="Observed value of the feature for this sample."
    )
    contribution: float = Field(
        description="Attribution / contribution score towards prediction (e.g. SHAP value or linear weight * x)."
    )


class LocalExplanationRecord(BaseModel):
    """Structured explanation for a single representative sample prediction."""
    sample_index: int = Field(description="Row index in dataset.")
    category: str = Field(
        description="Selection category (e.g. correct_positive, low_residual, representative_sample)."
    )
    prediction: Union[float, int, str, bool] = Field(description="Model prediction for this sample.")
    actual_value: Optional[Union[float, int, str, bool]] = Field(
        default=None,
        description="Ground truth target value if available."
    )
    base_value: Optional[float] = Field(
        default=None,
        description="Base value / expected value before feature contributions."
    )
    contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Ranked list of feature contributions for this sample."
    )


class ExplanationOutput(BaseModel):
    """Complete, structured output from the Explainability Engine."""
    dataset_id: str = Field(description="Dataset identifier or filename.")
    pipeline_id: str = Field(description="Registered pipeline identifier.")
    model_name: str = Field(description="Estimator class name.")
    task_type: str = Field(description="Task type: 'classification' or 'regression'.")
    metric: str = Field(description="Evaluation metric name (e.g. 'f1', 'rmse').")
    model_score: float = Field(description="Validation metric score achieved by model.")
    method: str = Field(
        description="Explanation strategy used ('shap_tree', 'linear_coefficients', 'native_tree', 'permutation_importance', 'unsupported')."
    )
    global_importance: List[FeatureImportanceRecord] = Field(
        default_factory=list,
        description="Ranked global feature importance list."
    )
    local_explanations: List[LocalExplanationRecord] = Field(
        default_factory=list,
        description="Representative local sample prediction explanations."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warnings or limitation notes generated during explanation."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional technical metadata (e.g., random_state, n_repeats, runtime_seconds)."
    )
