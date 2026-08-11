"""
Baseline & Ablation Execution Methods for GENESIS-AI Week 7 Research Evaluation (Hardened v1.3).

Implements comparison matrix for complete component isolation:
- Method A (FULL GENESIS-AI): DIP + Recommendation (Top-K=2) + Evolutionary Optimization
- Method B (WITHOUT DIP): Neutral Profile + Recommendation (Top-K=2) + Evolutionary Optimization (NO DIP)
- Method C (WITHOUT RECOMMENDATION): DIP Compatibility Filtering + Evolutionary Optimization (All Candidates)
- Method D (RECOMMENDATION ONLY): Top-K Recommended Default Pipelines (No GA)
- Method E (UNGUIDED BASELINE): Random Search across Task Search Space (Equal Budget, NO DIP)
"""

from typing import Dict, Any, Union, Optional, List
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

from backend.dataset.loader import load_csv
from backend.dataset.validator import validate_dataset
from backend.dataset.dip import generate_dip
from backend.recommendation.engine import RecommendationEngine
from backend.recommendation.registry import PipelineRegistry
from backend.recommendation.filters import apply_compatibility_filters
from backend.recommendation.schemas import NormalizedDIPSignals
from backend.recommendation.normalizer import normalize_dip_signals
from backend.optimization.schemas import OptimizationConfig, ChromosomeDict, GenerationHistory
from backend.optimization.optimizer import EvolutionaryOptimizer
from backend.optimization.chromosome import Chromosome
from backend.optimization.search_space import get_search_space, sample_random_hyperparameters
from backend.optimization.evaluator import build_sklearn_pipeline
from backend.optimization.population import generate_initial_population
from backend.optimization.selection import tournament_selection
from backend.optimization.crossover import crossover
from backend.optimization.mutation import mutate
from backend.optimization.cache import EvaluationCache
from backend.optimization.fitness import FitnessManager
from backend.evaluation.schemas import RawObservation, BenchmarkConfig
from backend.evaluation.datasets import DatasetManager
from backend.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    get_primary_metric_name
)


class BaselineRunnerError(Exception):
    """Exception raised during baseline execution failure."""
    pass


def create_neutral_dip(task_type: str, dataset_name: str = "dataset.csv") -> Dict[str, Any]:
    """
    Generate a neutral, dataset-agnostic DIP structure containing no empirical intelligence signals.
    Enables clean ablation of DIP intelligence through the exact same RecommendationEngine pathway.
    """
    return {
        "dataset": {
            "name": dataset_name,
            "num_rows": 100,
            "num_columns": 10,
            "memory_usage_mb": 0.1
        },
        "target": {
            "name": "target",
            "task_type": task_type,
            "num_classes": 2 if task_type.lower() == "classification" else None
        },
        "columns": {
            "numeric_count": 10,
            "categorical_count": 0,
            "datetime_count": 0,
            "column_types": {}
        },
        "missing_values": {
            "total_missing": 0,
            "missing_ratio": 0.0,
            "columns_with_missing": []
        },
        "imbalance": {
            "is_imbalanced": False,
            "class_imbalance_ratio": 1.0
        },
        "quality": {
            "outlier_ratio": 0.0,
            "duplicate_row_ratio": 0.0
        },
        "statistics": {
            "skewness_median": 0.0,
            "kurtosis_median": 0.0
        }
    }


class BaselineExecutor:
    """
    Executor for benchmark comparison methods A, B, C, D, E.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.registry = PipelineRegistry()

    def run_method(
        self,
        method_id: str,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        seed: int,
        dataset_name: str = "dataset.csv"
    ) -> RawObservation:
        """
        Dispatch execution to specific comparison method runner.
        """
        method_map = {
            "method_a_full_genesis": self.run_method_a,
            "method_b_without_dip": self.run_method_b,
            "method_c_without_recommendation": self.run_method_c,
            "method_d_recommendation_only": self.run_method_d,
            "method_e_unguided_baseline": self.run_method_e,
        }

        if method_id not in method_map:
            raise BaselineRunnerError(f"Unknown method_id: {method_id}")

        return method_map[method_id](df, target_column, task_type, seed, dataset_name)

    def run_method_a(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        seed: int,
        dataset_name: str = "dataset.csv"
    ) -> RawObservation:
        """
        Method A — FULL GENESIS-AI: DIP + Recommendation (Top-K=2) + Evolutionary Optimization
        """
        opt_config = OptimizationConfig(
            mode="genesis",
            top_k=self.config.top_k,
            population_size=self.config.population_size,
            generations=self.config.generations,
            max_evaluations=self.config.max_evaluations,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            pipeline_mutation_rate=self.config.pipeline_mutation_rate,
            random_state=seed
        )
        optimizer = EvolutionaryOptimizer(config=opt_config, registry=self.registry)
        res = optimizer.optimize(df, target_column=target_column, dataset_name=dataset_name)

        primary_metric = get_primary_metric_name(task_type)
        score = float(res.test_performance.get(primary_metric, 0.0))

        return RawObservation(
            dataset=dataset_name,
            task_type=task_type,
            method="method_a_full_genesis",
            seed=seed,
            metric=primary_metric,
            score=score,
            candidate_evaluations=res.evaluations_used,
            unique_evaluations=res.unique_evaluations,
            runtime_seconds=res.runtime_seconds,
            best_configuration={
                "pipeline_id": res.best_pipeline_id,
                "pipeline_name": res.best_pipeline_name,
                "hyperparameters": res.best_hyperparameters
            },
            status="success",
            error=None
        )

    def _run_ga_over_candidate_pool(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        seed: int,
        candidate_pool: List[Any],
        method_name: str,
        dataset_name: str
    ) -> RawObservation:
        """
        Helper executing GA Evolutionary Optimization over a pre-selected candidate pool
        WITHOUT calling generate_dip or DIP modules.
        """
        start_time = time.perf_counter()

        ds_mgr = DatasetManager()
        X_train, X_val, X_test, y_train, y_val, y_test = ds_mgr.create_splits(
            df=df,
            target_column=target_column,
            task_type=task_type,
            seed=seed,
            test_ratio=self.config.test_ratio,
            val_ratio=self.config.val_ratio
        )

        candidate_pipeline_ids = [p.pipeline_id for p in candidate_pool]
        rng = random.Random(seed)

        population = generate_initial_population(
            candidate_pipelines=candidate_pool,
            pop_size=self.config.population_size,
            rng=rng
        )

        cache = EvaluationCache()
        fitness_manager = FitnessManager(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=task_type,
            max_evaluations=self.config.max_evaluations,
            cache=cache,
            random_state=seed
        )

        overall_best_chrom: Optional[Chromosome] = None
        overall_best_fitness: float = float("-inf")

        for gen in range(1, self.config.generations + 1):
            if fitness_manager.evaluations_used >= self.config.max_evaluations:
                break

            fitnesses, _ = fitness_manager.evaluate_population(population)
            best_gen_fitness = max(fitnesses)
            best_gen_idx = int(np.argmax(fitnesses))

            if best_gen_fitness > overall_best_fitness:
                overall_best_fitness = best_gen_fitness
                overall_best_chrom = population[best_gen_idx].copy()

            if fitness_manager.evaluations_used >= self.config.max_evaluations:
                break

            sorted_indices = np.argsort(fitnesses)[::-1]
            elites = [population[idx].copy() for idx in sorted_indices[:2]]

            parents = tournament_selection(
                population=population,
                fitnesses=fitnesses,
                tournament_size=3,
                num_select=self.config.population_size,
                rng=rng
            )

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

        if overall_best_chrom is None:
            overall_best_chrom = population[0]

        # Isolated Test Evaluation
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)

        best_pipeline_model = build_sklearn_pipeline(
            pipeline_id=overall_best_chrom.pipeline_id,
            hyperparameters=overall_best_chrom.hyperparameters,
            X_sample=X_train_val,
            random_state=seed
        )
        best_pipeline_model.fit(X_train_val, y_train_val)
        y_test_pred = best_pipeline_model.predict(X_test)

        primary_metric = get_primary_metric_name(task_type)
        if task_type.lower() == "classification":
            metrics = calculate_classification_metrics(y_test, y_test_pred)
        else:
            metrics = calculate_regression_metrics(y_test, y_test_pred)

        score = metrics[primary_metric]
        elapsed = round(time.perf_counter() - start_time, 2)

        pipe_meta = self.registry.get_pipeline_by_id(overall_best_chrom.pipeline_id)
        pipe_name = pipe_meta.name if pipe_meta else overall_best_chrom.pipeline_id

        return RawObservation(
            dataset=dataset_name,
            task_type=task_type,
            method=method_name,
            seed=seed,
            metric=primary_metric,
            score=score,
            candidate_evaluations=fitness_manager.evaluations_used,
            unique_evaluations=cache.unique_evaluations,
            runtime_seconds=elapsed,
            best_configuration={
                "pipeline_id": overall_best_chrom.pipeline_id,
                "pipeline_name": pipe_name,
                "hyperparameters": overall_best_chrom.hyperparameters
            },
            status="success",
            error=None
        )

    def run_method_b(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        seed: int,
        dataset_name: str = "dataset.csv"
    ) -> RawObservation:
        """
        Method B — WITHOUT DIP: Equivalent Recommendation Pathway + Evolutionary Optimization
        NO DIP profiling. Generates a neutral/non-DIP profile and queries the exact same
        RecommendationEngine pathway as Method A, then runs GA evolution.
        """
        neutral_dip = create_neutral_dip(task_type=task_type, dataset_name=dataset_name)

        rec_engine = RecommendationEngine(registry=self.registry)
        rec_report = rec_engine.recommend_from_dip(neutral_dip, top_k=self.config.top_k)

        top_rec_ids = [r.pipeline_id for r in rec_report.recommendations]
        all_candidates = self.registry.get_all_pipelines()
        candidate_pool = [p for p in all_candidates if p.pipeline_id in top_rec_ids]

        if not candidate_pool:
            raise BaselineRunnerError("Zero candidate pipelines returned by Recommendation Engine for Method B.")

        return self._run_ga_over_candidate_pool(
            df=df,
            target_column=target_column,
            task_type=task_type,
            seed=seed,
            candidate_pool=candidate_pool,
            method_name="method_b_without_dip",
            dataset_name=dataset_name
        )

    def run_method_c(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        seed: int,
        dataset_name: str = "dataset.csv"
    ) -> RawObservation:
        """
        Method C — WITHOUT RECOMMENDATION: DIP + Evolutionary Optimization (All Compatible Candidates)
        """
        opt_config = OptimizationConfig(
            mode="baseline",
            top_k=10,
            population_size=self.config.population_size,
            generations=self.config.generations,
            max_evaluations=self.config.max_evaluations,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            pipeline_mutation_rate=self.config.pipeline_mutation_rate,
            random_state=seed
        )
        optimizer = EvolutionaryOptimizer(config=opt_config, registry=self.registry)
        res = optimizer.optimize(df, target_column=target_column, dataset_name=dataset_name)

        primary_metric = get_primary_metric_name(task_type)
        score = float(res.test_performance.get(primary_metric, 0.0))

        return RawObservation(
            dataset=dataset_name,
            task_type=task_type,
            method="method_c_without_recommendation",
            seed=seed,
            metric=primary_metric,
            score=score,
            candidate_evaluations=res.evaluations_used,
            unique_evaluations=res.unique_evaluations,
            runtime_seconds=res.runtime_seconds,
            best_configuration={
                "pipeline_id": res.best_pipeline_id,
                "pipeline_name": res.best_pipeline_name,
                "hyperparameters": res.best_hyperparameters
            },
            status="success",
            error=None
        )

    def run_method_d(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        seed: int,
        dataset_name: str = "dataset.csv"
    ) -> RawObservation:
        """
        Method D — RECOMMENDATION ONLY: Default Hyperparameter Selection from Top-K Pool (No GA)
        Queries RecommendationEngine using exact same DIP profile and top_k pool as Method A.
        Evaluates default hyperparameters for each candidate in the Top-K pool on Validation split,
        picks the best default candidate model, and evaluates once on isolated Test split.
        """
        start_time = time.perf_counter()

        ds_mgr = DatasetManager()
        X_train, X_val, X_test, y_train, y_val, y_test = ds_mgr.create_splits(
            df=df,
            target_column=target_column,
            task_type=task_type,
            seed=seed,
            test_ratio=self.config.test_ratio,
            val_ratio=self.config.val_ratio
        )

        dip_dict = generate_dip(df, target_column=target_column, dataset_name=dataset_name)
        rec_engine = RecommendationEngine(registry=self.registry)
        rec_report = rec_engine.recommend_from_dip(dip_dict, top_k=self.config.top_k)

        top_rec_ids = [r.pipeline_id for r in rec_report.recommendations]

        best_pipeline_id: Optional[str] = None
        best_default_hp: Optional[Dict[str, Any]] = None
        best_val_score = float("-inf")
        evaluations_count = 0

        for pipe_id in top_rec_ids:
            search_grid = get_search_space(pipe_id)
            default_hp = {k: v[0] for k, v in search_grid.items()}
            evaluations_count += 1

            try:
                model = build_sklearn_pipeline(
                    pipeline_id=pipe_id,
                    hyperparameters=default_hp,
                    X_sample=X_train,
                    random_state=seed
                )
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)

                if task_type.lower() == "classification":
                    val_fit = calculate_classification_metrics(y_val, y_val_pred)["f1"]
                else:
                    val_fit = -calculate_regression_metrics(y_val, y_val_pred)["rmse"]

                if val_fit > best_val_score:
                    best_val_score = val_fit
                    best_pipeline_id = pipe_id
                    best_default_hp = default_hp
            except Exception:
                continue

        if best_pipeline_id is None:
            best_pipeline_id = top_rec_ids[0]
            search_grid = get_search_space(best_pipeline_id)
            best_default_hp = {k: v[0] for k, v in search_grid.items()}

        # Isolated Test Set Evaluation
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)

        final_model = build_sklearn_pipeline(
            pipeline_id=best_pipeline_id,
            hyperparameters=best_default_hp,
            X_sample=X_train_val,
            random_state=seed
        )
        final_model.fit(X_train_val, y_train_val)
        y_test_pred = final_model.predict(X_test)

        primary_metric = get_primary_metric_name(task_type)
        if task_type.lower() == "classification":
            metrics = calculate_classification_metrics(y_test, y_test_pred)
        else:
            metrics = calculate_regression_metrics(y_test, y_test_pred)

        score = metrics[primary_metric]
        elapsed = round(time.perf_counter() - start_time, 2)

        pipe_meta = self.registry.get_pipeline_by_id(best_pipeline_id)
        pipe_name = pipe_meta.name if pipe_meta else best_pipeline_id

        return RawObservation(
            dataset=dataset_name,
            task_type=task_type,
            method="method_d_recommendation_only",
            seed=seed,
            metric=primary_metric,
            score=score,
            candidate_evaluations=evaluations_count,
            unique_evaluations=evaluations_count,
            runtime_seconds=elapsed,
            best_configuration={
                "pipeline_id": best_pipeline_id,
                "pipeline_name": pipe_name,
                "hyperparameters": best_default_hp
            },
            status="success",
            error=None
        )

    def run_method_e(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        seed: int,
        dataset_name: str = "dataset.csv"
    ) -> RawObservation:
        """
        Method E — UNGUIDED BASELINE SEARCH: Random Search over Task Search Space (Equal Budget, NO DIP)
        NO DIP profiling or DIP compatibility filters. Samples candidates matching task_type uniformly.
        """
        start_time = time.perf_counter()

        ds_mgr = DatasetManager()
        X_train, X_val, X_test, y_train, y_val, y_test = ds_mgr.create_splits(
            df=df,
            target_column=target_column,
            task_type=task_type,
            seed=seed,
            test_ratio=self.config.test_ratio,
            val_ratio=self.config.val_ratio
        )

        all_candidates = self.registry.get_all_pipelines()
        task_candidates = [p for p in all_candidates if p.task.lower() == task_type.lower()]

        if not task_candidates:
            raise BaselineRunnerError("Zero candidate pipelines matching task type for Method E.")

        rng = random.Random(seed)
        primary_metric = get_primary_metric_name(task_type)

        best_chrom: Optional[Chromosome] = None
        best_val_fitness = float("-inf")
        evaluations_used = 0
        evaluated_hashes = set()

        for _ in range(self.config.max_evaluations):
            pipe = rng.choice(task_candidates)
            hp = sample_random_hyperparameters(pipe.pipeline_id, rng=rng)
            chrom = Chromosome(pipeline_id=pipe.pipeline_id, hyperparameters=hp)
            h = chrom.get_hash()
            evaluated_hashes.add(h)
            evaluations_used += 1

            try:
                model = build_sklearn_pipeline(
                    pipeline_id=chrom.pipeline_id,
                    hyperparameters=chrom.hyperparameters,
                    X_sample=X_train,
                    random_state=seed
                )
                model.fit(X_train, y_train)
                y_val_pred = model.predict(X_val)

                if task_type.lower() == "classification":
                    val_fit = calculate_classification_metrics(y_val, y_val_pred)["f1"]
                else:
                    val_rmse = calculate_regression_metrics(y_val, y_val_pred)["rmse"]
                    val_fit = -val_rmse

                if val_fit > best_val_fitness:
                    best_val_fitness = val_fit
                    best_chrom = chrom
            except Exception:
                continue

        if best_chrom is None:
            pipe_fallback = rng.choice(task_candidates)
            hp_fallback = sample_random_hyperparameters(pipe_fallback.pipeline_id, rng=rng)
            best_chrom = Chromosome(pipeline_id=pipe_fallback.pipeline_id, hyperparameters=hp_fallback)

        # Isolated Test Set Evaluation
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)

        best_model = build_sklearn_pipeline(
            pipeline_id=best_chrom.pipeline_id,
            hyperparameters=best_chrom.hyperparameters,
            X_sample=X_train_val,
            random_state=seed
        )
        best_model.fit(X_train_val, y_train_val)
        y_test_pred = best_model.predict(X_test)

        if task_type.lower() == "classification":
            test_metrics = calculate_classification_metrics(y_test, y_test_pred)
        else:
            test_metrics = calculate_regression_metrics(y_test, y_test_pred)

        score = test_metrics[primary_metric]
        elapsed = round(time.perf_counter() - start_time, 2)

        pipe_meta = self.registry.get_pipeline_by_id(best_chrom.pipeline_id)
        pipe_name = pipe_meta.name if pipe_meta else best_chrom.pipeline_id

        return RawObservation(
            dataset=dataset_name,
            task_type=task_type,
            method="method_e_unguided_baseline",
            seed=seed,
            metric=primary_metric,
            score=score,
            candidate_evaluations=evaluations_used,
            unique_evaluations=len(evaluated_hashes),
            runtime_seconds=elapsed,
            best_configuration={
                "pipeline_id": best_chrom.pipeline_id,
                "pipeline_name": pipe_name,
                "hyperparameters": best_chrom.hyperparameters
            },
            status="success",
            error=None
        )
