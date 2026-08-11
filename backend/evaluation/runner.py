"""
Benchmark Execution Runner for GENESIS-AI Week 7 Research Evaluation.

Orchestrates multi-dataset, multi-method, multi-seed experiment matrix execution,
enforces error isolation, progress tracking, and raw observation output persistence.
"""

from typing import List, Dict, Any, Optional
import json
import time
import traceback
import logging
from pathlib import Path

from backend.evaluation.schemas import BenchmarkConfig, RawObservation, DatasetSpec
from backend.evaluation.configuration import get_default_benchmark_config
from backend.evaluation.datasets import DatasetManager
from backend.evaluation.baselines import BaselineExecutor
from backend.evaluation.protocols import FairnessProtocol

logger = logging.getLogger("genesis.evaluation.runner")


class BenchmarkRunner:
    """
    High-level orchestrator for executing the Week 7 Research Evaluation Benchmark.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or get_default_benchmark_config()
        self.dataset_manager = DatasetManager(specs=self.config.datasets)
        self.baseline_executor = BaselineExecutor(config=self.config)
        self.protocol = FairnessProtocol(config=self.config)

    def run_benchmark(self, verbose: bool = True) -> List[RawObservation]:
        """
        Execute the full benchmark evaluation matrix across all datasets, methods, and seeds.

        Args:
            verbose: If True, prints progress updates.

        Returns:
            List of RawObservation instances containing empirical results.
        """
        is_valid, errors = self.protocol.validate_config()
        if not is_valid:
            raise ValueError(f"Benchmark configuration violates fairness protocol: {errors}")

        observations: List[RawObservation] = []
        total_runs = len(self.config.datasets) * len(self.config.methods) * len(self.config.seeds)
        completed = 0

        if verbose:
            print("=" * 100)
            print(f"GENESIS-AI WEEK 7 RESEARCH EVALUATION BENCHMARK RUNNER")
            print(f"Total Execution Matrix: {len(self.config.datasets)} Datasets x {len(self.config.methods)} Methods x {len(self.config.seeds)} Seeds = {total_runs} Runs")
            print("=" * 100)

        start_all_time = time.perf_counter()

        for ds_spec in self.config.datasets:
            filename = ds_spec.filename
            target_col = ds_spec.target
            task_type = ds_spec.task_type

            df = self.dataset_manager.load_dataset(filename)

            for method_id in self.config.methods:
                for seed in self.config.seeds:
                    completed += 1
                    if verbose:
                        print(f"[{completed}/{total_runs}] Running Dataset='{filename}' | Method='{method_id}' | Seed={seed}...", end="", flush=True)

                    try:
                        obs = self.baseline_executor.run_method(
                            method_id=method_id,
                            df=df,
                            target_column=target_col,
                            task_type=task_type,
                            seed=seed,
                            dataset_name=filename
                        )
                        observations.append(obs)
                        if verbose:
                            print(f" DONE ({obs.metric}={obs.score:.4f}, evals={obs.candidate_evaluations}, time={obs.runtime_seconds:.2f}s)")
                    except Exception as e:
                        err_msg = f"{str(e)}\n{traceback.format_exc()}"
                        logger.error(f"Failed benchmark run for {filename} {method_id} seed {seed}: {err_msg}")
                        fallback_score = -1.0 if task_type == "classification" else 999999.0
                        obs = RawObservation(
                            dataset=filename,
                            task_type=task_type,
                            method=method_id,
                            seed=seed,
                            metric="f1" if task_type == "classification" else "rmse",
                            score=fallback_score,
                            candidate_evaluations=0,
                            unique_evaluations=0,
                            runtime_seconds=0.0,
                            best_configuration={},
                            status="failed",
                            error=str(e)
                        )
                        observations.append(obs)
                        if verbose:
                            print(f" FAILED ({str(e)})")

        elapsed_all = round(time.perf_counter() - start_all_time, 2)
        if verbose:
            print("=" * 100)
            print(f"Benchmark execution completed in {elapsed_all}s. Total observations collected: {len(observations)}")
            print("=" * 100)

        # Save raw results to JSON file
        output_path = Path(self.config.output_benchmark_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_data = [obs.model_dump() for obs in observations]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)

        if verbose:
            print(f"Saved raw benchmark results to: {output_path}")

        return observations
