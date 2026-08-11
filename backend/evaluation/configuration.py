"""
Centralized Configuration Manager for GENESIS-AI Week 7 Research Evaluation.

Provides standardized experiment parameters, dataset registries, comparison method definitions,
fixed random seeds, evaluation budgets, and file artifact paths.
"""

from typing import List, Dict, Any
from pathlib import Path
from backend.evaluation.schemas import BenchmarkConfig, DatasetSpec

BASE_DIR = Path(__file__).parent.parent.parent
DATASETS_DIR = BASE_DIR / "data" / "test_datasets"

DEFAULT_DATASET_SPECS = [
    DatasetSpec(
        filename="01_numerical_classification.csv",
        target="target",
        task_type="classification",
        description="Clean numerical binary classification dataset."
    ),
    DatasetSpec(
        filename="02_categorical_heavy.csv",
        target="subscribed",
        task_type="classification",
        description="Categorical-heavy binary classification dataset."
    ),
    DatasetSpec(
        filename="03_missing_values.csv",
        target="label",
        task_type="classification",
        description="Classification dataset containing missing feature values."
    ),
    DatasetSpec(
        filename="04_imbalanced_classification.csv",
        target="is_fraud",
        task_type="classification",
        description="Highly class-imbalanced binary classification dataset."
    ),
    DatasetSpec(
        filename="05_regression.csv",
        target="price",
        task_type="regression",
        description="Continuous target house price regression dataset."
    ),
]

METHOD_METADATA: Dict[str, Dict[str, Any]] = {
    "method_a_full_genesis": {
        "code": "A",
        "name": "Full GENESIS-AI",
        "has_dip": True,
        "has_recommendation": True,
        "has_optimization": True,
        "description": "DIP + Recommendation (Top-K=2) + Evolutionary Optimization"
    },
    "method_b_without_dip": {
        "code": "B",
        "name": "Without DIP",
        "has_dip": False,
        "has_recommendation": True,
        "has_optimization": True,
        "description": "Recommendation (Uniform/Static Ranking) + Evolutionary Optimization"
    },
    "method_c_without_recommendation": {
        "code": "C",
        "name": "Without Recommendation",
        "has_dip": True,
        "has_recommendation": False,
        "has_optimization": True,
        "description": "DIP Compatibility Filtering + Evolutionary Optimization (All Candidates)"
    },
    "method_d_recommendation_only": {
        "code": "D",
        "name": "Recommendation Only",
        "has_dip": True,
        "has_recommendation": True,
        "has_optimization": False,
        "description": "Top Recommended Pipeline with Default Hyperparameters (No GA)"
    },
    "method_e_unguided_baseline": {
        "code": "E",
        "name": "Unguided Baseline",
        "has_dip": False,
        "has_recommendation": False,
        "has_optimization": False,
        "description": "Random Search over Compatible Search Space (Equal Evaluation Budget)"
    },
}

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]


def get_default_benchmark_config() -> BenchmarkConfig:
    """Construct and return default research evaluation configuration."""
    return BenchmarkConfig(
        benchmark_version="1.0",
        datasets=DEFAULT_DATASET_SPECS,
        methods=list(METHOD_METADATA.keys()),
        seeds=DEFAULT_SEEDS,
        max_evaluations=200,
        top_k=2,
        population_size=20,
        generations=10,
        crossover_rate=0.80,
        mutation_rate=0.10,
        pipeline_mutation_rate=0.10,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        output_benchmark_json=str(BASE_DIR / "experiments" / "week7_benchmark_results.json"),
        output_summary_csv=str(BASE_DIR / "experiments" / "week7_summary.csv"),
        output_ablation_json=str(BASE_DIR / "experiments" / "week7_ablation_results.json")
    )
