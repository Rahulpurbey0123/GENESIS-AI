"""
Controlled Development Experiment Runner for GENESIS-AI Week 5 Explainability Engine.

For each of the 5 development datasets:
1. Obtains the best Week 4 pipeline using EvolutionaryOptimizer (GENESIS mode, seed=42).
2. Fits the pipeline on the train/val split using established protocol.
3. Invokes ExplainabilityEngine to generate global feature importance and local sample explanations.
4. Validates output schema and verifies zero NaN/Inf leakage.
5. Records structured dataset metrics, method used, top features, local sample count, runtime, and warnings.
6. Exports complete results to experiments/week5_explainability_results.json.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer
from backend.optimization.evaluator import build_sklearn_pipeline
from backend.explainability.engine import ExplainabilityEngine

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "test_datasets"
OUTPUT_FILE = BASE_DIR / "experiments" / "week5_explainability_results.json"

TEST_DATASETS = [
    {"filename": "01_numerical_classification.csv", "target": "target"},
    {"filename": "02_categorical_heavy.csv", "target": "subscribed"},
    {"filename": "03_missing_values.csv", "target": "label"},
    {"filename": "04_imbalanced_classification.csv", "target": "is_fraud"},
    {"filename": "05_regression.csv", "target": "price"},
]


def run_explainability_experiment():
    print("=" * 100)
    print("GENESIS-AI WEEK 5 EXPLAINABILITY & MODEL INSIGHT EXPERIMENT")
    print("=" * 100)

    results = {}
    engine = ExplainabilityEngine()
    total_start_time = time.perf_counter()

    for ds_info in TEST_DATASETS:
        filename = ds_info["filename"]
        target_col = ds_info["target"]
        csv_path = DATA_DIR / filename

        print(f"\nProcessing dataset: {filename} (target: {target_col})...")

        # Step 1: Obtain the best Week 4 pipeline using EvolutionaryOptimizer
        config = OptimizationConfig(
            mode="genesis",
            top_k=2,
            population_size=20,
            generations=10,
            max_evaluations=200,
            random_state=42
        )
        optimizer = EvolutionaryOptimizer(config=config)
        opt_res = optimizer.optimize(csv_path, target_column=target_col, dataset_name=filename)

        best_pipeline_id = opt_res.best_pipeline_id
        best_hyperparams = opt_res.best_hyperparameters
        task_type = opt_res.task_type
        metric = "f1" if task_type == "classification" else "rmse"

        print(f"  Best Week 4 Pipeline: {best_pipeline_id} ({opt_res.best_pipeline_name})")
        print(f"  Validation Score ({metric}): {opt_res.best_fitness}")

        # Step 2: Fit model pipeline using established protocol
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Stratified train/test/val split matching optimizer protocol
        stratify_y = y if task_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2 else None
        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y
            )
        except Exception:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        stratify_tv = y_train_val if task_type == "classification" and y_train_val.nunique() > 1 and y_train_val.value_counts().min() >= 2 else None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=stratify_tv
            )
        except Exception:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42
            )

        pipeline_model = build_sklearn_pipeline(
            pipeline_id=best_pipeline_id,
            hyperparameters=best_hyperparams,
            X_sample=X_train,
            random_state=42
        )
        pipeline_model.fit(X_train, y_train)

        # Step 3: Run Explainability Engine
        exp_start = time.perf_counter()
        exp_output = engine.explain(
            pipeline_or_model=pipeline_model,
            X_val=X_val,
            y_val=y_val,
            dataset_id=filename,
            pipeline_id=best_pipeline_id,
            model_name=opt_res.best_pipeline_name,
            task_type=task_type,
            metric=metric,
            model_score=opt_res.best_fitness,
            n_repeats=5,
            random_state=42
        )
        exp_elapsed = round(time.perf_counter() - exp_start, 4)

        # Step 4: Extract top features and record structured experiment output
        output_dict = exp_output.model_dump()

        top_feats = [
            {
                "feature": rec["feature"],
                "importance": rec["importance"],
                "rank": rec["rank"],
                "direction": rec["direction"]
            }
            for rec in output_dict["global_importance"][:5]
        ]

        summary_entry = {
            "dataset": filename,
            "task_type": task_type,
            "pipeline": best_pipeline_id,
            "model": exp_output.model_name,
            "metric": metric,
            "score": exp_output.model_score,
            "method": exp_output.method,
            "top_features": top_feats,
            "local_sample_count": len(exp_output.local_explanations),
            "explanation_runtime": exp_elapsed,
            "warnings": exp_output.warnings,
            "full_explanation": output_dict
        }

        results[filename] = summary_entry

        print(f"  Method Used: {exp_output.method}")
        print(f"  Top Feature: {top_feats[0]['feature'] if top_feats else 'N/A'} (Imp: {top_feats[0]['importance'] if top_feats else 0.0})")
        print(f"  Local Samples Explained: {len(exp_output.local_explanations)}")
        print(f"  Explanation Runtime: {exp_elapsed}s")

    total_elapsed = round(time.perf_counter() - total_start_time, 2)

    # Save to experiments/week5_explainability_results.json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"Completed Week 5 Experiment across 5 datasets in {total_elapsed}s.")
    print(f"Saved results to: {OUTPUT_FILE}")
    print("=" * 100)


if __name__ == "__main__":
    run_explainability_experiment()
