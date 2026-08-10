"""
Structured Pydantic schemas and dataclasses for Evolutionary Optimization Engine v1.0.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class OptimizationConfig(BaseModel):
    """Configurable parameters for Evolutionary Pipeline Optimization Engine."""
    mode: str = "genesis"  # "genesis" | "baseline"
    top_k: int = Field(5, ge=1)
    population_size: int = Field(20, ge=2)
    generations: int = Field(10, ge=1)
    max_evaluations: int = Field(200, ge=1)
    crossover_rate: float = Field(0.80, ge=0.0, le=1.0)
    mutation_rate: float = Field(0.10, ge=0.0, le=1.0)          # Hyperparameter mutation rate
    pipeline_mutation_rate: float = Field(0.10, ge=0.0, le=1.0) # Model-family mutation rate
    elite_size: int = Field(2, ge=0)
    tournament_size: int = Field(3, ge=1)
    random_state: int = 42
    test_size: float = Field(0.20, gt=0.0, lt=1.0)
    val_size: float = Field(0.20, gt=0.0, lt=1.0)
    metric: Optional[str] = None  # Optional custom metric ("f1", "accuracy", "rmse", "mae", "r2")

    model_config = ConfigDict(extra="ignore")


class ChromosomeDict(BaseModel):
    """Pydantic model representing an individual chromosome in the population."""
    pipeline_id: str
    hyperparameters: Dict[str, Any]

    model_config = ConfigDict(extra="ignore")


class GenerationHistory(BaseModel):
    """Execution history metrics for a single GA generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    evaluations_used: int
    cache_hits: int


class OptimizationResult(BaseModel):
    """Structured result object returned by Evolutionary Pipeline Optimization Engine."""
    mode: str
    task_type: str
    random_state: int
    population_size: int
    generations: int
    max_evaluations: int
    evaluations_used: int
    unique_evaluations: int
    cache_hits: int
    candidate_count_before: int
    candidate_count_after: int
    candidate_space_reduction: float
    candidate_pipeline_ids: List[str]
    best_pipeline_id: str
    best_pipeline_name: str
    best_hyperparameters: Dict[str, Any]
    best_fitness: float  # Validation set fitness score
    test_performance: Optional[Dict[str, float]] = None  # Isolated test set metrics evaluated post-GA
    generation_found: int
    runtime_seconds: float
    history: List[GenerationHistory]
    warnings: List[str]
    baseline_best_pipeline: Optional[str] = None
    baseline_best_in_genesis_top_k: Optional[bool] = None
    optimization_version: str = "1.2"
