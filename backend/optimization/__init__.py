"""
GENESIS-AI Evolutionary Pipeline Optimization Engine Package v1.0.

Provides Genetic Algorithm optimization for machine learning pipelines and hyperparameters.
"""

from backend.optimization.schemas import (
    OptimizationConfig,
    OptimizationResult,
    GenerationHistory,
    ChromosomeDict,
)
from backend.optimization.search_space import (
    SEARCH_SPACES,
    get_search_space,
    sample_random_hyperparameters,
    validate_hyperparameters,
)
from backend.optimization.chromosome import Chromosome
from backend.optimization.cache import EvaluationCache
from backend.optimization.evaluator import (
    build_estimator,
    build_sklearn_pipeline,
    evaluate_chromosome,
)
from backend.optimization.population import generate_initial_population
from backend.optimization.selection import tournament_selection
from backend.optimization.crossover import crossover
from backend.optimization.mutation import mutate
from backend.optimization.fitness import FitnessManager
from backend.optimization.optimizer import (
    EvolutionaryOptimizer,
    EvolutionaryOptimizerError,
)
from backend.optimization.modes import run_genesis_mode, run_baseline_mode

__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "GenerationHistory",
    "ChromosomeDict",
    "SEARCH_SPACES",
    "get_search_space",
    "sample_random_hyperparameters",
    "validate_hyperparameters",
    "Chromosome",
    "EvaluationCache",
    "build_estimator",
    "build_sklearn_pipeline",
    "evaluate_chromosome",
    "generate_initial_population",
    "tournament_selection",
    "crossover",
    "mutate",
    "FitnessManager",
    "EvolutionaryOptimizer",
    "EvolutionaryOptimizerError",
    "run_genesis_mode",
    "run_baseline_mode",
]
