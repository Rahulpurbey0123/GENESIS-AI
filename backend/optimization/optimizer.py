"""
Core Evolutionary Optimizer for GENESIS-AI Optimization Engine v1.0.

Orchestrates data splitting, candidate pool selection (GENESIS vs BASELINE), GA generation loop,
elitism, budget enforcement, evaluation caching, and isolated test set evaluation.
"""

from typing import Dict, Any, Union, Optional, List, Tuple, Callable
import time
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

from backend.dataset.loader import load_csv
from backend.dataset.validator import validate_dataset
from backend.dataset.cleaner import clean_dataset_for_ml
from backend.dataset.dip import generate_dip
from backend.recommendation.engine import RecommendationEngine
from backend.recommendation.registry import PipelineRegistry
from backend.recommendation.filters import apply_compatibility_filters
from backend.recommendation.normalizer import normalize_dip_signals

from backend.optimization.schemas import (
    OptimizationConfig,
    OptimizationResult,
    GenerationHistory,
    ChromosomeDict
)
from backend.optimization.chromosome import Chromosome
from backend.optimization.search_space import validate_hyperparameters
from backend.optimization.cache import EvaluationCache
from backend.optimization.population import generate_initial_population
from backend.optimization.selection import tournament_selection
from backend.optimization.crossover import crossover
from backend.optimization.mutation import mutate
from backend.optimization.fitness import FitnessManager
from backend.optimization.evaluator import build_sklearn_pipeline, evaluate_chromosome

logger = logging.getLogger("genesis.optimization.optimizer")


class EvolutionaryOptimizerError(Exception):
    """Custom exception raised for optimization failures."""
    pass


class EvolutionaryOptimizer:
    """
    Genetic Algorithm Evolutionary Pipeline Optimization Engine.
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        registry: Optional[PipelineRegistry] = None
    ):
        self.config = config or OptimizationConfig()
        self.registry = registry or PipelineRegistry()

    def optimize(
        self,
        file_source_or_df: Union[str, Path, bytes, pd.DataFrame],
        target_column: str,
        dataset_name: str = "dataset.csv",
        progress_callback: Optional[Callable[[int, int, int, float, float], None]] = None,
        evaluate_test: bool = True
    ) -> OptimizationResult:

        """
        Execute Evolutionary Pipeline Optimization in GENESIS mode or BASELINE mode.

        Args:
            file_source_or_df: CSV filepath, bytes, or pandas DataFrame.
            target_column: Exact target column name.
            dataset_name: Display name or filename.

        Returns:
            Structured OptimizationResult object.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []

        # Step 1: Load and Validate Dataset
        if isinstance(file_source_or_df, pd.DataFrame):
            df = file_source_or_df.copy()
        else:
            df = load_csv(file_source_or_df, filename=dataset_name)

        val_report = validate_dataset(df, target_column)
        target_col = val_report["target_column"]

        # Clean dataset for ML (removes missing/NaN/inf target rows and replaces feature infs)
        df = clean_dataset_for_ml(df, target_column=target_col)

        # Step 2: Extract DIP v1.1 Profile
        dip_dict = generate_dip(df, target_column=target_col, dataset_name=dataset_name)
        task_type = dip_dict["target"]["task_type"].lower()

        # Step 3: Train / Validation / Test Dataset Splitting
        from backend.dataset.contract import get_canonical_data_split
        X, y, feature_names, actual_target_col, identifier_cols = get_canonical_data_split(
            df, target_column=target_col, exclude_identifiers=True
        )



        # Stratified split for classification if possible
        stratify_y = y if task_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2 else None

        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_y
            )
        except Exception:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

        # Split Train_Val into Train (60%) and Validation (20%)
        val_ratio_adjusted = self.config.val_size / (1.0 - self.config.test_size)
        stratify_tv = y_train_val if task_type == "classification" and y_train_val.nunique() > 1 and y_train_val.value_counts().min() >= 2 else None

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_ratio_adjusted,
                random_state=self.config.random_state,
                stratify=stratify_tv
            )
        except Exception:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_ratio_adjusted,
                random_state=self.config.random_state
            )

        # Step 4: Candidate Search Space Selection (GENESIS vs BASELINE mode)
        rec_engine = RecommendationEngine(registry=self.registry)
        signals = normalize_dip_signals(dip_dict)
        all_registry_candidates = self.registry.get_all_pipelines()
        compatible_candidates, filter_warns = apply_compatibility_filters(all_registry_candidates, signals)
        warnings.extend(filter_warns)

        if not compatible_candidates:
            raise EvolutionaryOptimizerError("Zero compatible pipelines found for dataset task.")

        candidate_count_before = len(compatible_candidates)

        if self.config.mode.lower() == "genesis":
            rec_report = rec_engine.recommend_from_dip(dip_dict, top_k=self.config.top_k)
            top_rec_ids = [r.pipeline_id for r in rec_report.recommendations]
            candidate_pool = [p for p in compatible_candidates if p.pipeline_id in top_rec_ids]
            if not candidate_pool:
                candidate_pool = compatible_candidates
        else:  # BASELINE mode
            candidate_pool = compatible_candidates

        candidate_count_after = len(candidate_pool)
        candidate_pipeline_ids = [p.pipeline_id for p in candidate_pool]

        if candidate_count_before > 0:
            candidate_space_reduction = round((candidate_count_before - candidate_count_after) / candidate_count_before, 4)
        else:
            candidate_space_reduction = 0.0

        # Step 5: Initialize GA Seed and Initial Population
        rng = random.Random(self.config.random_state)
        population = generate_initial_population(
            candidate_pipelines=candidate_pool,
            pop_size=self.config.population_size,
            rng=rng
        )

        # Step 6: Initialize Cache & Fitness Manager
        cache = EvaluationCache()
        fitness_manager = FitnessManager(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=task_type,
            max_evaluations=self.config.max_evaluations,
            cache=cache,
            random_state=self.config.random_state,
            metric_name=self.config.metric
        )

        overall_best_chrom: Optional[Chromosome] = None
        overall_best_fitness: float = float("-inf")
        generation_found: int = 1
        history: List[GenerationHistory] = []

        # Step 7: GA Generation Loop
        for gen in range(1, self.config.generations + 1):
            if fitness_manager.evaluations_used >= self.config.max_evaluations:
                warnings.append(f"Max evaluations budget ({self.config.max_evaluations}) reached at generation {gen}.")
                break

            # Evaluate population fitness on Validation split
            fitnesses, metrics_list = fitness_manager.evaluate_population(population)

            best_gen_fitness = max(fitnesses)
            valid_fits = [f for f in fitnesses if f != float("-inf")]
            mean_gen_fitness = float(np.mean(valid_fits)) if valid_fits else float("-inf")

            # Update overall best chromosome
            best_gen_idx = int(np.argmax(fitnesses))
            if best_gen_fitness > overall_best_fitness:
                overall_best_fitness = best_gen_fitness
                overall_best_chrom = population[best_gen_idx].copy()
                generation_found = gen

            history.append(
                GenerationHistory(
                    generation=gen,
                    best_fitness=round(best_gen_fitness, 4) if best_gen_fitness != float("-inf") else float("-inf"),
                    mean_fitness=round(mean_gen_fitness, 4) if mean_gen_fitness != float("-inf") else float("-inf"),
                    evaluations_used=fitness_manager.evaluations_used,
                    cache_hits=cache.cache_hits,
                )
            )

            if progress_callback:
                elapsed_now = round(time.perf_counter() - start_time, 2)
                cur_best = round(overall_best_fitness, 4) if overall_best_fitness != float("-inf") else 0.0
                progress_callback(gen, self.config.generations, fitness_manager.evaluations_used, cur_best, elapsed_now)

            # Check evaluation budget before breeding next generation
            if fitness_manager.evaluations_used >= self.config.max_evaluations:
                break

            # Elitism: Preserve best elite_size individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            elites = [population[idx].copy() for idx in sorted_indices[:self.config.elite_size]]

            # Parent Selection
            parents = tournament_selection(
                population=population,
                fitnesses=fitnesses,
                tournament_size=self.config.tournament_size,
                num_select=self.config.population_size,
                rng=rng
            )

            # Crossover & Mutation to create offspring
            next_population = list(elites)
            parent_idx = 0
            while len(next_population) < self.config.population_size:
                p1 = parents[parent_idx % len(parents)]
                p2 = parents[(parent_idx + 1) % len(parents)]
                parent_idx += 2

                off1, off2 = crossover(p1, p2, crossover_rate=self.config.crossover_rate, rng=rng)
                off1 = mutate(
                    off1,
                    mutation_rate=self.config.mutation_rate,
                    pipeline_mutation_rate=self.config.pipeline_mutation_rate,
                    allowed_pipeline_ids=candidate_pipeline_ids,
                    rng=rng
                )
                off2 = mutate(
                    off2,
                    mutation_rate=self.config.mutation_rate,
                    pipeline_mutation_rate=self.config.pipeline_mutation_rate,
                    allowed_pipeline_ids=candidate_pipeline_ids,
                    rng=rng
                )

                next_population.append(off1)
                if len(next_population) < self.config.population_size:
                    next_population.append(off2)

            population = next_population[:self.config.population_size]

        if overall_best_chrom is None or overall_best_fitness == float("-inf"):
            raise EvolutionaryOptimizerError(
                "All candidate pipeline evaluations failed. Please check dataset feature and target validity."
            )


        # Step 8: Isolated Test Set Evaluation
        test_performance: Dict[str, float] = {}
        if evaluate_test:
            try:
                best_pipeline_model = build_sklearn_pipeline(
                    pipeline_id=overall_best_chrom.pipeline_id,
                    hyperparameters=overall_best_chrom.hyperparameters,
                    X_sample=X_train_val,
                    random_state=self.config.random_state
                )
                # Fit on combined Train + Validation splits
                best_pipeline_model.fit(X_train_val, y_train_val)
                y_test_pred = best_pipeline_model.predict(X_test)

                if task_type == "classification":
                    test_acc = float(accuracy_score(y_test, y_test_pred))
                    test_f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
                    test_performance = {"f1": round(test_f1, 4), "accuracy": round(test_acc, 4)}
                else:
                    test_mse = float(mean_squared_error(y_test, y_test_pred))
                    test_rmse = float(np.sqrt(test_mse))
                    test_mae = float(mean_absolute_error(y_test, y_test_pred))
                    test_r2 = float(r2_score(y_test, y_test_pred))
                    test_performance = {
                        "rmse": round(test_rmse, 4),
                        "mae": round(test_mae, 4),
                        "r2": round(test_r2, 4)
                    }
            except Exception as e:
                logger.warning(f"Final test set evaluation failed: {str(e)}")
                test_performance = {"error": -1.0}


        end_time = time.perf_counter()
        elapsed_sec = round(end_time - start_time, 2)

        pipeline_meta = self.registry.get_pipeline_by_id(overall_best_chrom.pipeline_id)
        best_name = pipeline_meta.name if pipeline_meta else overall_best_chrom.pipeline_id

        # Step 9: Construct Structured Optimization Result
        return OptimizationResult(
            mode=self.config.mode.lower(),
            task_type=task_type,
            random_state=self.config.random_state,
            population_size=self.config.population_size,
            generations=len(history),
            max_evaluations=self.config.max_evaluations,
            evaluations_used=fitness_manager.evaluations_used,
            unique_evaluations=cache.unique_evaluations,
            cache_hits=cache.cache_hits,
            candidate_count_before=candidate_count_before,
            candidate_count_after=candidate_count_after,
            candidate_space_reduction=candidate_space_reduction,
            candidate_pipeline_ids=candidate_pipeline_ids,
            best_pipeline_id=overall_best_chrom.pipeline_id,
            best_pipeline_name=best_name,
            best_hyperparameters=overall_best_chrom.hyperparameters,
            best_fitness=round(overall_best_fitness, 4) if overall_best_fitness != float("-inf") else float("-inf"),
            test_performance=test_performance,
            generation_found=generation_found,
            runtime_seconds=elapsed_sec,
            history=history,
            warnings=warnings,
        )
