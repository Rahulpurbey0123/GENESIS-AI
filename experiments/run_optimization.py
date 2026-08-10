"""
Controlled Multi-Seed & Multi-Top-K Experimental Validation Runner for GENESIS-AI Week 4.2.

Runs paired GENESIS vs BASELINE experiments across 5 validation datasets, 5 random seeds (42, 123, 456, 789, 2024),
and multiple Top-K values (2, 3, 4, 5). Calculates baseline winner retention diagnostics.
Exports raw results to experiments/week4_optimization_results.json.
"""

import json
import time
from pathlib import Path
from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "test_datasets"
OUTPUT_FILE = BASE_DIR / "experiments" / "week4_optimization_results.json"

TEST_DATASETS = [
    {"filename": "01_numerical_classification.csv", "target": "target"},
    {"filename": "02_categorical_heavy.csv", "target": "subscribed"},
    {"filename": "03_missing_values.csv", "target": "label"},
    {"filename": "04_imbalanced_classification.csv", "target": "is_fraud"},
    {"filename": "05_regression.csv", "target": "price"},
]

SEEDS = [42, 123, 456, 789, 2024]
TOP_K_VALUES = [2, 3, 4, 5]


def run_optimization_experiments():
    results = {}

    print("=" * 100)
    print("GENESIS-AI WEEK 4.2 CONTROLLED MULTI-SEED EXPERIMENTAL VALIDATION")
    print("=" * 100)

    total_runs = 0
    start_all_time = time.perf_counter()

    for ds_info in TEST_DATASETS:
        filename = ds_info["filename"]
        target_col = ds_info["target"]
        csv_path = DATA_DIR / filename

        results[filename] = {}

        for seed in SEEDS:
            seed_key = f"seed_{seed}"
            results[filename][seed_key] = {}

            # 1. Run BASELINE Mode First (Control Group)
            base_config = OptimizationConfig(
                mode="baseline",
                top_k=10,
                population_size=20,
                generations=10,
                max_evaluations=200,
                crossover_rate=0.80,
                mutation_rate=0.10,
                pipeline_mutation_rate=0.10,
                elite_size=2,
                tournament_size=3,
                random_state=seed
            )
            base_optimizer = EvolutionaryOptimizer(config=base_config)
            base_res = base_optimizer.optimize(csv_path, target_column=target_col, dataset_name=filename)

            baseline_best_pipeline_id = base_res.best_pipeline_id
            compat_count = base_res.candidate_count_before

            base_dict = base_res.model_dump()
            base_dict["baseline_best_pipeline"] = baseline_best_pipeline_id
            base_dict["baseline_best_in_genesis_top_k"] = True
            base_dict["search_space_version"] = "1.0"
            base_dict["recommendation_version"] = "1.1"
            base_dict["optimizer_version"] = "1.2"

            results[filename][seed_key]["baseline"] = base_dict
            total_runs += 1

            print(f"[{filename} | Seed {seed}] BASELINE: Best={baseline_best_pipeline_id}, ValFit={base_res.best_fitness}, TestFit={base_res.test_performance}, Compat={compat_count}")

            # 2. Run GENESIS Mode across valid Top-K values
            for k in TOP_K_VALUES:
                # Top-K cannot exceed total compatible candidates available
                if k > compat_count and k > 2:
                    continue

                effective_k = min(k, compat_count)

                gen_config = OptimizationConfig(
                    mode="genesis",
                    top_k=effective_k,
                    population_size=20,
                    generations=10,
                    max_evaluations=200,
                    crossover_rate=0.80,
                    mutation_rate=0.10,
                    pipeline_mutation_rate=0.10,
                    elite_size=2,
                    tournament_size=3,
                    random_state=seed
                )
                gen_optimizer = EvolutionaryOptimizer(config=gen_config)
                gen_res = gen_optimizer.optimize(csv_path, target_column=target_col, dataset_name=filename)

                best_in_top_k = baseline_best_pipeline_id in gen_res.candidate_pipeline_ids

                gen_dict = gen_res.model_dump()
                gen_dict["baseline_best_pipeline"] = baseline_best_pipeline_id
                gen_dict["baseline_best_in_genesis_top_k"] = best_in_top_k
                gen_dict["search_space_version"] = "1.0"
                gen_dict["recommendation_version"] = "1.1"
                gen_dict["optimizer_version"] = "1.2"

                topk_key = f"genesis_topk_{k}"
                results[filename][seed_key][topk_key] = gen_dict
                total_runs += 1

                print(
                    f"[{filename} | Seed {seed}] GENESIS Top-K={k}: Best={gen_res.best_pipeline_id}, "
                    f"ValFit={gen_res.best_fitness}, TestFit={gen_res.test_performance}, "
                    f"Red={gen_res.candidate_space_reduction * 100:.1f}%, WinnerRetained={best_in_top_k}"
                )

    end_all_time = time.perf_counter()
    total_elapsed = round(end_all_time - start_all_time, 2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"Executed {total_runs} total experiment runs in {total_elapsed}s.")
    print(f"Raw validation experiment results saved to: {OUTPUT_FILE}")
    print("=" * 100)


if __name__ == "__main__":
    run_optimization_experiments()
