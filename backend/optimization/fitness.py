"""
Fitness Manager and Budget Enforcer for Evolutionary Pipeline Optimization Engine v1.0.

Orchestrates fitness evaluation, budget tracking, and evaluation caching.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import pandas as pd

from backend.optimization.chromosome import Chromosome
from backend.optimization.cache import EvaluationCache
from backend.optimization.evaluator import evaluate_chromosome

logger = logging.getLogger("genesis.optimization.fitness")


class FitnessManager:
    """
    Coordinates population fitness evaluation while enforcing hard evaluation budget limits
    and cache lookups.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        task_type: str,
        max_evaluations: int = 200,
        cache: Optional[EvaluationCache] = None,
        random_state: int = 42,
        metric_name: Optional[str] = None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.task_type = task_type
        self.max_evaluations = max_evaluations
        self.cache = cache or EvaluationCache()
        self.random_state = random_state
        self.metric_name = metric_name

        self.evaluations_used: int = 0

    def evaluate_population(
        self,
        population: List[Chromosome]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Evaluate fitness for every chromosome in the population.

        Checks evaluation budget. If max_evaluations is reached, un-evaluated individuals
        receive worst-case default fitness (-999.0).

        Args:
            population: List of Chromosome objects to evaluate.

        Returns:
            Tuple of (fitness_scores_list, detailed_metrics_list).
        """
        fitnesses: List[float] = []
        metrics_list: List[Dict[str, Any]] = []

        for chrom in population:
            cached_res = self.cache.get(chrom)
            if cached_res is not None:
                fitnesses.append(cached_res["fitness"])
                metrics_list.append(cached_res["metrics"])
                continue

            # Cache miss - check budget limit
            if self.evaluations_used >= self.max_evaluations:
                logger.warning(
                    f"Max evaluations budget ({self.max_evaluations}) reached. Assigning fallback fitness."
                )
                fitnesses.append(float("-inf"))
                metrics_list.append({"budget_exceeded": 1.0, "status": "failed", "fitness": float("-inf")})
                continue

            # Perform actual pipeline training and validation
            fitness, metrics = evaluate_chromosome(
                chromosome=chrom,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                task_type=self.task_type,
                random_state=self.random_state,
                metric_name=self.metric_name
            )

            self.evaluations_used += 1
            self.cache.put(chrom, fitness, metrics)

            fitnesses.append(fitness)
            metrics_list.append(metrics)

        return fitnesses, metrics_list
