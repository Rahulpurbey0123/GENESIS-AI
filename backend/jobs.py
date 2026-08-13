"""
Asynchronous background job manager for long-running GENESIS-AI optimization experiments.
Executes DIP generation, Recommendation filtering, Evolutionary Optimization, and Explainability analysis.
"""

import time
import logging
import threading
import uuid
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from backend.dataset.contract import get_canonical_data_split


from backend.database import DatabaseService
from backend.dataset.loader import load_csv
from backend.dataset.cleaner import clean_dataset_for_ml

from backend.dataset.dip import generate_dip
from backend.recommendation.engine import RecommendationEngine
from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer
from backend.optimization.evaluator import build_sklearn_pipeline
from backend.explainability.engine import ExplainabilityEngine

logger = logging.getLogger("genesis.jobs")

executor = ThreadPoolExecutor(max_workers=4)


class ExperimentJob:
    """Task runner for a single optimization experiment."""

    @staticmethod
    def run(experiment_id: str, dataset_id: str, target_column: str, config_dict: Dict[str, Any]) -> None:
        logger.info(f"Starting background experiment job '{experiment_id}' for dataset '{dataset_id}'")
        try:
            # Step 1: Fetch dataset from DB/disk
            dataset_record = DatabaseService.get_dataset(dataset_id)
            if not dataset_record:
                raise ValueError(f"Dataset '{dataset_id}' not found in database.")

            filepath = dataset_record["filepath"]
            dataset_name = dataset_record["name"]

            with open(filepath, "rb") as f:
                csv_bytes = f.read()

            df = load_csv(csv_bytes, filename=dataset_name)
            df = clean_dataset_for_ml(df, target_column=target_column)


            # Step 2: Ensure DIP profile is available
            dip_profile = DatabaseService.get_dip_profile(dataset_id)
            if not dip_profile:
                dip_profile = generate_dip(csv_bytes, target_column=target_column, dataset_name=dataset_name)
                DatabaseService.save_dip_profile(dataset_id, target_column, dip_profile)

            # Step 3: Run Recommendation Engine to measure search space reduction
            rec_engine = RecommendationEngine()
            rec_report = rec_engine.recommend(csv_bytes, target_column=target_column, dataset_name=dataset_name, top_k=config_dict.get("top_k", 5))

            search_space_reduction = rec_report.search_space_reduction
            task_type = rec_report.task_type.lower()

            # Step 4: Configure and Run Evolutionary Optimizer
            config = OptimizationConfig(
                mode=config_dict.get("mode", "genesis"),
                top_k=config_dict.get("top_k", 2),
                population_size=config_dict.get("population_size", 20),
                generations=config_dict.get("generations", 10),
                max_evaluations=config_dict.get("max_evaluations", 200),
                mutation_rate=config_dict.get("mutation_rate", 0.10),
                pipeline_mutation_rate=config_dict.get("pipeline_mutation_rate", 0.10),
                random_state=config_dict.get("random_state", 42)
            )

            # Define real GA generation progress callback with history tracking
            ga_history: List[Dict[str, Any]] = []

            def on_ga_progress(gen: int, total_gens: int, evals: int, best_s: float, elapsed: float):
                if not any(h["gen"] == gen for h in ga_history):
                    ga_history.append({"gen": gen, "best_score": round(best_s, 4)})
                DatabaseService.update_experiment_progress(
                    experiment_id=experiment_id,
                    status="RUNNING",
                    progress={
                        "current_generation": gen,
                        "max_generations": total_gens,
                        "evaluated_pipelines": evals,
                        "best_score": round(best_s, 4),
                        "runtime": round(elapsed, 2),
                        "search_space_reduction": round(search_space_reduction, 4),
                        "history": ga_history
                    }
                )

            optimizer = EvolutionaryOptimizer(config=config)
            opt_result = optimizer.optimize(
                csv_bytes,
                target_column=target_column,
                dataset_name=dataset_name,
                progress_callback=on_ga_progress,
                evaluate_test=False
            )


            best_score = float(opt_result.best_fitness)
            evaluations_count = getattr(opt_result, 'evaluations_used', 0)
            runtime_sec = float(getattr(opt_result, 'runtime_seconds', 0.0))

            best_pipeline_dict = {
                "id": getattr(opt_result, 'best_pipeline_id', 'best_pipeline'),
                "model_name": getattr(opt_result, 'best_pipeline_name', 'Best Estimator'),
                "task_type": rec_report.task_type,
                "preprocessing": getattr(opt_result, 'best_hyperparameters', {}),
                "hyperparameters": getattr(opt_result, 'best_hyperparameters', {}),
                "fitness": best_score
            }

            # Step 5: Isolated Test Set Evaluation & Real Scikit-Learn Metrics
            X, y, feature_names, actual_target_col, identifier_cols = get_canonical_data_split(
                df, target_column=target_column, exclude_identifiers=True
            )

            stratify_y = y if task_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2 else None
            try:
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=config.random_state, stratify=stratify_y
                )
            except Exception:
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=config.random_state
                )

            best_pipeline_model = build_sklearn_pipeline(
                pipeline_id=opt_result.best_pipeline_id,
                hyperparameters=opt_result.best_hyperparameters,
                X_sample=X_train_val,
                random_state=config.random_state
            )
            best_pipeline_model.fit(X_train_val, y_train_val)
            y_test_pred = best_pipeline_model.predict(X_test)

            metrics: Dict[str, Any] = {}
            evaluation_plots: Dict[str, Any] = {
                "confusion_matrix": None,
                "roc_curve": None,
                "residuals": None
            }

            if task_type == "classification":
                acc_val = round(float(accuracy_score(y_test, y_test_pred)), 4)
                f1_val = round(float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)), 4)
                prec_val = round(float(precision_score(y_test, y_test_pred, average="macro", zero_division=0)), 4)
                rec_val = round(float(recall_score(y_test, y_test_pred, average="macro", zero_division=0)), 4)
                bal_acc_val = round(float(balanced_accuracy_score(y_test, y_test_pred)), 4)

                roc_auc_val = None
                pr_auc_val = None
                roc_curve_val = None

                # Compute ROC-AUC, PR-AUC & ROC Curve if probabilities or decision_function are available
                try:
                    if hasattr(best_pipeline_model, "predict_proba"):
                        y_test_prob = best_pipeline_model.predict_proba(X_test)
                        if len(np.unique(y_test)) == 2:
                            roc_auc_val = round(float(roc_auc_score(y_test, y_test_prob[:, 1])), 4)
                            fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1], pos_label=best_pipeline_model.classes_[1])
                            roc_curve_val = {
                                "fpr": [round(float(x), 4) for x in fpr.tolist()],
                                "tpr": [round(float(x), 4) for x in tpr.tolist()]
                            }
                            # Calculate real Precision-Recall AUC (PR-AUC)
                            p_pts, r_pts, _ = precision_recall_curve(y_test, y_test_prob[:, 1], pos_label=best_pipeline_model.classes_[1])
                            pr_auc_val = round(float(auc(r_pts, p_pts)), 4)
                        else:
                            roc_auc_val = round(float(roc_auc_score(y_test, y_test_prob, multi_class="ovr")), 4)
                    elif hasattr(best_pipeline_model, "decision_function"):
                        y_test_scores = best_pipeline_model.decision_function(X_test)
                        if len(np.unique(y_test)) == 2:
                            roc_auc_val = round(float(roc_auc_score(y_test, y_test_scores)), 4)
                            fpr, tpr, _ = roc_curve(y_test, y_test_scores, pos_label=best_pipeline_model.classes_[1])
                            roc_curve_val = {
                                "fpr": [round(float(x), 4) for x in fpr.tolist()],
                                "tpr": [round(float(x), 4) for x in tpr.tolist()]
                            }
                except Exception as e:
                    logger.info(f"ROC/PR calculation unavailable for this model: {str(e)}")

                metrics = {
                    "accuracy": acc_val,
                    "f1": f1_val,
                    "precision": prec_val,
                    "recall": rec_val,
                    "balanced_accuracy": bal_acc_val,
                    "roc_auc": roc_auc_val,
                    "pr_auc": pr_auc_val
                }


                # Real Confusion Matrix from y_test and y_test_pred
                try:
                    cm = confusion_matrix(y_test, y_test_pred).tolist()
                    evaluation_plots["confusion_matrix"] = cm
                except Exception as e:
                    logger.info(f"Confusion matrix calculation unavailable: {str(e)}")

                evaluation_plots["roc_curve"] = roc_curve_val

            else:  # Regression
                mae_val = round(float(mean_absolute_error(y_test, y_test_pred)), 4)
                mse_val = float(mean_squared_error(y_test, y_test_pred))
                rmse_val = round(float(np.sqrt(mse_val)), 4)
                r2_val = round(float(r2_score(y_test, y_test_pred)), 4)

                metrics = {
                    "mae": mae_val,
                    "rmse": rmse_val,
                    "r2": r2_val
                }

                # Real Regression Residuals from y_test and y_test_pred
                try:
                    evaluation_plots["residuals"] = {
                        "y_true": [round(float(x), 4) for x in y_test.values[:50].tolist()],
                        "y_pred": [round(float(x), 4) for x in y_test_pred[:50].tolist()]
                    }
                except Exception as e:
                    logger.info(f"Residual plot calculation unavailable: {str(e)}")

            efficiency = {
                "runtime_seconds": round(runtime_sec, 2),
                "pipelines_evaluated": evaluations_count,
                "generations": config.generations,
                "search_space_reduction": round(search_space_reduction, 4)
            }

            DatabaseService.save_experiment_results(
                experiment_id=experiment_id,
                best_pipeline=best_pipeline_dict,
                metrics=metrics,
                efficiency=efficiency
            )

            # Step 6: Real Explainability Engine Generation (Week 5)
            explain_engine = ExplainabilityEngine()
            explanation_out = explain_engine.explain(
                pipeline_or_model=best_pipeline_model,
                X_val=X_test,
                y_val=y_test,
                dataset_id=dataset_name,
                pipeline_id=opt_result.best_pipeline_id,
                task_type=task_type
            )

            global_importance_list = [
                {
                    "feature": str(fi.feature),
                    "importance": round(float(fi.importance), 4),
                    "rank": fi.rank
                }
                for fi in explanation_out.global_importance
            ]

            shap_summary = []
            is_shap_method = (explanation_out.method == "shap_tree")
            for fi in explanation_out.global_importance[:15]:
                item = {
                    "feature": str(fi.feature),
                    "summary": f"Feature '{fi.feature}' rank #{fi.rank} with importance {fi.importance:.4f} via {explanation_out.method}."
                }
                if is_shap_method:
                    item["mean_shap_value"] = round(float(fi.importance), 4)
                else:
                    item["importance"] = round(float(fi.importance), 4)
                shap_summary.append(item)

            local_importance_data = {
                "sample_index": 0,
                "features": global_importance_list[:5]
            }
            if explanation_out.local_explanations:
                local_importance_data["local_samples"] = [
                    loc.model_dump() for loc in explanation_out.local_explanations
                ]

            shap_payload = {
                "summary": shap_summary,
                "method": explanation_out.method,
                "disclaimer": "Feature importance indicates predictive contribution/association in the model. It does not establish causality."
            }

            DatabaseService.save_experiment_explanations(
                experiment_id=experiment_id,
                shap_data=shap_payload,
                global_importance={"features": global_importance_list, "method": explanation_out.method},
                local_importance=local_importance_data,
                evaluation_plots=evaluation_plots
            )

            # Final status update
            DatabaseService.update_experiment_progress(
                experiment_id=experiment_id,
                status="COMPLETED",
                progress={
                    "current_generation": config.generations,
                    "max_generations": config.generations,
                    "evaluated_pipelines": evaluations_count,
                    "best_score": round(best_score, 4),
                    "runtime": round(runtime_sec, 2),
                    "search_space_reduction": round(search_space_reduction, 4),
                    "history": ga_history
                }
            )
            logger.info(f"Experiment '{experiment_id}' completed successfully.")

        except Exception as e:
            logger.error(f"Experiment '{experiment_id}' failed: {str(e)}", exc_info=True)
            existing_exp = DatabaseService.get_experiment(experiment_id)
            current_progress = existing_exp.get("progress", {}) if existing_exp else {}

            failed_progress = {
                "current_generation": current_progress.get("current_generation", 0),
                "max_generations": current_progress.get("max_generations", config_dict.get("generations", 10)),
                "evaluated_pipelines": current_progress.get("evaluated_pipelines", 0),
                "best_score": current_progress.get("best_score", None),
                "runtime": current_progress.get("runtime", 0.0),
                "search_space_reduction": current_progress.get("search_space_reduction", 0.0),
                "history": current_progress.get("history", [])
            }

            DatabaseService.update_experiment_progress(
                experiment_id=experiment_id,
                status="FAILED",
                progress=failed_progress,
                error_message=str(e)
            )



class JobManager:
    """Service to submit and track experiment background tasks."""

    @staticmethod
    def submit_experiment(
        dataset_id: str,
        target_column: str,
        mode: str = "genesis",
        config_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config_dict = config_dict or {}
        config_dict["mode"] = mode
        experiment_id = f"exp_{uuid.uuid4().hex[:10]}"

        dataset_record = DatabaseService.get_dataset(dataset_id)
        dataset_name = dataset_record["name"] if dataset_record else "dataset.csv"

        exp_record = DatabaseService.create_experiment(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            target_column=target_column,
            mode=mode,
            config=config_dict
        )

        executor.submit(ExperimentJob.run, experiment_id, dataset_id, target_column, config_dict)
        return exp_record
