"""
Mode Helpers for GENESIS-AI Evolutionary Pipeline Optimization Engine v1.1.

Provides convenience wrappers to run GENESIS mode and BASELINE mode optimization.
"""

from typing import Union, Optional, Dict, Any
from pathlib import Path
import pandas as pd

from backend.optimization.schemas import OptimizationConfig, OptimizationResult
from backend.optimization.optimizer import EvolutionaryOptimizer


def run_genesis_mode(
    file_source_or_df: Union[str, Path, bytes, pd.DataFrame],
    target_column: str,
    top_k: int = 2,
    population_size: int = 20,
    generations: int = 10,
    max_evaluations: int = 200,
    mutation_rate: float = 0.10,
    pipeline_mutation_rate: float = 0.10,
    random_state: int = 42,
    dataset_name: str = "dataset.csv"
) -> OptimizationResult:
    """
    Run Evolutionary Optimization in GENESIS Mode.

    Restricts GA candidate search space to Top-K candidate recommendations from Week 3 engine.
    """
    config = OptimizationConfig(
        mode="genesis",
        top_k=top_k,
        population_size=population_size,
        generations=generations,
        max_evaluations=max_evaluations,
        mutation_rate=mutation_rate,
        pipeline_mutation_rate=pipeline_mutation_rate,
        random_state=random_state
    )
    optimizer = EvolutionaryOptimizer(config=config)
    return optimizer.optimize(file_source_or_df, target_column=target_column, dataset_name=dataset_name)


def run_baseline_mode(
    file_source_or_df: Union[str, Path, bytes, pd.DataFrame],
    target_column: str,
    population_size: int = 20,
    generations: int = 10,
    max_evaluations: int = 200,
    mutation_rate: float = 0.10,
    pipeline_mutation_rate: float = 0.10,
    random_state: int = 42,
    dataset_name: str = "dataset.csv"
) -> OptimizationResult:
    """
    Run Evolutionary Optimization in BASELINE Mode.

    Uses all compatible candidate pipelines from Stage 1 filtering as the control group.
    """
    config = OptimizationConfig(
        mode="baseline",
        top_k=10,
        population_size=population_size,
        generations=generations,
        max_evaluations=max_evaluations,
        mutation_rate=mutation_rate,
        pipeline_mutation_rate=pipeline_mutation_rate,
        random_state=random_state
    )
    optimizer = EvolutionaryOptimizer(config=config)
    return optimizer.optimize(file_source_or_df, target_column=target_column, dataset_name=dataset_name)
