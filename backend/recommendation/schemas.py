"""
Structured Pydantic schemas and dataclasses for Recommendation Engine v1.1.
"""

import math
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


class PipelineStep(BaseModel):
    """Metadata for an individual step in a recommended ML pipeline."""
    step_id: str
    step_type: str  # "imputer", "scaler", "encoder", "model"
    name: str
    class_name: str


class PipelineMetadata(BaseModel):
    """Metadata describing a candidate ML pipeline in the registry."""
    pipeline_id: str
    name: str
    task: str  # "classification" | "regression"
    model_family: str  # "linear", "tree_ensemble", "svm", "knn"
    model_name: str
    steps: List[PipelineStep]
    requires_scaling: bool = False
    supports_class_weight: bool = False
    handles_categorical_natively: bool = False
    nonlinear: bool = True
    computational_cost: str = "medium"  # "low", "medium", "high"
    small_data_suitable: bool = True
    large_data_suitable: bool = True
    high_dimensional_suitable: bool = True

    model_config = ConfigDict(extra="ignore")


class NormalizedDIPSignals(BaseModel):
    """Derived and normalized signals extracted from DIP v1.1 for recommendation."""
    task_type: str
    rows: int
    columns: int
    feature_count: int
    numeric_features: int
    categorical_features: int
    binary_features: int
    numeric_ratio: float
    categorical_ratio: float
    binary_ratio: float
    total_missing: int
    missing_rate: float
    feature_missingness: float
    target_missingness: float
    missingness_level: str  # "none", "low", "moderate", "high"
    target_missing_flag: bool
    outlier_rate: float
    mean_absolute_skewness: float
    imbalance_ratio: float
    minority_percentage: Optional[float] = None
    imbalance_severity: str  # "none", "moderate", "severe"
    dimensionality_ratio: float  # feature_count / rows
    high_dimensional_flag: bool
    dataset_size_category: str  # "small", "medium", "large"
    categorical_heavy_flag: bool
    numerical_heavy_flag: bool
    complexity_score: float
    complexity_label: str

    model_config = ConfigDict(extra="ignore")


class RecommendationReason(BaseModel):
    """Structured rule explanation containing rule identifier and machine-generated heuristic reason."""
    rule_id: str
    reason: str


class ScoringWeights(BaseModel):
    """Configurable weights for multi-criteria recommendation suitability scoring (must sum to 1.0)."""
    task: float = Field(0.20, ge=0.0, le=1.0)
    dataset_size: float = Field(0.15, ge=0.0, le=1.0)
    feature_type: float = Field(0.20, ge=0.0, le=1.0)
    missingness: float = Field(0.10, ge=0.0, le=1.0)
    imbalance: float = Field(0.10, ge=0.0, le=1.0)
    dimensionality: float = Field(0.10, ge=0.0, le=1.0)
    computational: float = Field(0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "ScoringWeights":
        weights = [
            self.task,
            self.dataset_size,
            self.feature_type,
            self.missingness,
            self.imbalance,
            self.dimensionality,
            self.computational,
        ]
        for w in weights:
            if math.isnan(w) or math.isinf(w):
                raise ValueError("Recommendation scoring weights cannot contain NaN or infinity.")
            if w < 0.0:
                raise ValueError("Recommendation scoring weights must be non-negative.")

        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Recommendation scoring weights must sum to 1.0. Current sum: {total_weight:.6f}"
            )
        return self


class ThresholdConfig(BaseModel):
    """Configurable engineering thresholds for recommendation rules."""
    small_dataset_max_rows: int = 1000
    large_dataset_min_rows: int = 50000
    high_dimensionality_ratio: float = 0.10
    high_feature_count: int = 100
    categorical_heavy_ratio: float = 0.40
    numerical_heavy_ratio: float = 0.70
    moderate_missing_rate: float = 0.05
    high_missing_rate: float = 0.20
    moderate_imbalance_ratio: float = 1.5
    severe_imbalance_ratio: float = 4.0
    high_outlier_rate: float = 0.10
    high_skewness: float = 1.5


class RecommendationItem(BaseModel):
    """Structure for a single recommended pipeline entry in the report."""
    rank: int
    pipeline_id: str
    name: str
    model_family: str
    model_name: str
    score: float
    sub_scores: Dict[str, float]
    reasons: List[RecommendationReason]
    pipeline_metadata: Dict[str, Any]


class RecommendationReport(BaseModel):
    """Structured response report returned by the Recommendation Engine."""
    recommendation_version: str = "1.1"
    task_type: str
    candidate_count_before: int
    candidate_count_after_filtering: int
    filtering_reduction: float
    top_k: int
    recommended_count: int
    top_k_selection_ratio: float
    search_space_reduction: float  # Deprecated alias equal to filtering_reduction
    recommendations: List[RecommendationItem]
    warnings: List[str]
    dataset_summary: Dict[str, Any]
