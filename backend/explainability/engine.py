"""
Core Explainability Engine for GENESIS-AI Week 5 Explainability & Model Insight Engine.

Orchestrates post-hoc model explainability, strategy selection, global feature ranking,
local prediction explanations, validation, safety invariants, and structured JSON output.
"""

from typing import Dict, List, Optional, Any, Union
import time
import logging
import numpy as np
import pandas as pd

from backend.explainability.schemas import ExplanationOutput, FeatureImportanceRecord, LocalExplanationRecord
from backend.explainability.registry import ExplanationRegistry
from backend.explainability.validators import (
    validate_fitted_model,
    validate_feature_names,
    validate_importance_values,
    validate_no_nan_inf,
)
from backend.explainability.shap_explainer import HAS_SHAP, explain_shap_tree, get_local_shap_contributions
from backend.explainability.native_importance import (
    explain_linear_coefficients,
    get_local_linear_contributions,
    explain_native_tree_importance,
)
from backend.explainability.permutation import explain_permutation_importance

import importlib
_global_mod = importlib.import_module("backend.explainability.global")
format_global_importance = _global_mod.format_global_importance
from backend.explainability.local import (
    generate_local_explanations,
    select_representative_samples_classification,
    select_representative_samples_regression,
)

logger = logging.getLogger("genesis.explainability.engine")


def clean_feature_name(name: str) -> str:
    """Clean sklearn ColumnTransformer prefixes like 'num__' or 'cat__'."""
    s = str(name)
    if s.startswith("num__"):
        return s[5:]
    if s.startswith("cat__"):
        return s[5:]
    return s


class ExplainabilityEngine:
    """
    Post-hoc Explainability & Model Insight Engine.
    """

    def __init__(self, registry: Optional[ExplanationRegistry] = None):
        self.registry = registry or ExplanationRegistry()

    def explain(
        self,
        pipeline_or_model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        dataset_id: str = "dataset.csv",
        pipeline_id: str = "custom_pipeline",
        model_name: Optional[str] = None,
        task_type: str = "classification",
        metric: str = "f1",
        model_score: float = 0.0,
        n_repeats: int = 5,
        random_state: int = 42
    ) -> ExplanationOutput:
        """
        Generate structured global and local explanations for a fitted model/pipeline.

        Args:
            pipeline_or_model: Fitted scikit-learn Pipeline or estimator object.
            X_val: Held-out validation feature DataFrame.
            y_val: Held-out validation target Series.
            dataset_id: Display dataset identifier.
            pipeline_id: Registered candidate pipeline identifier.
            model_name: Display estimator model name.
            task_type: "classification" or "regression".
            metric: Evaluation metric name.
            model_score: Metric score achieved on validation split.
            n_repeats: Permutation importance repeats count.
            random_state: Seed for reproducibility.

        Returns:
            Structured ExplanationOutput object.
        """
        start_time = time.perf_counter()
        warnings: List[str] = []

        # Determine default research scoring metric
        scoring_metric = "f1_macro" if task_type.lower() == "classification" else "neg_root_mean_squared_error"

        # Step 1: Validate fitted model
        is_fitted, fit_err = validate_fitted_model(pipeline_or_model)
        if not is_fitted:
            return ExplanationOutput(
                dataset_id=dataset_id,
                pipeline_id=pipeline_id,
                model_name=model_name or "Unknown",
                task_type=task_type,
                metric=metric,
                model_score=model_score,
                method="unsupported",
                global_importance=[],
                local_explanations=[],
                warnings=[f"Model validation failed: {fit_err}"],
                metadata={
                    "scoring_metric": scoring_metric,
                    "random_state": random_state,
                    "runtime_seconds": 0.0
                }
            )

        # Step 2: Separate preprocessing pipeline step from model estimator
        if hasattr(pipeline_or_model, "named_steps"):
            preprocessor = pipeline_or_model.named_steps.get("preprocessor", None)
            model = pipeline_or_model.named_steps.get("model", pipeline_or_model)
        else:
            preprocessor = None
            model = pipeline_or_model

        actual_model_name = model_name or model.__class__.__name__

        # Step 3: Extract preprocessed feature matrix & feature names
        if not isinstance(X_val, pd.DataFrame):
            X_val_df = pd.DataFrame(X_val)
        else:
            X_val_df = X_val.copy()

        if preprocessor is not None:
            try:
                X_trans = preprocessor.transform(X_val_df)
                if hasattr(preprocessor, "get_feature_names_out"):
                    raw_names = list(preprocessor.get_feature_names_out())
                    feature_names = [clean_feature_name(n) for n in raw_names]
                else:
                    feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]
            except Exception as e:
                warnings.append(f"Preprocessor transform failed ({str(e)}). Using raw features.")
                X_trans = X_val_df.to_numpy()
                feature_names = list(X_val_df.columns)
        else:
            X_trans = X_val_df.to_numpy()
            feature_names = list(X_val_df.columns)

        feature_names, fn_warns = validate_feature_names(X_trans, feature_names)
        warnings.extend(fn_warns)

        # Step 4: Determine explanation strategy
        primary_strategy = self.registry.get_strategy(model)

        global_recs: List[FeatureImportanceRecord] = []
        local_recs: List[LocalExplanationRecord] = []
        executed_method = primary_strategy

        # Step 5: Execute strategy cleanly according to valid local capability
        if primary_strategy == "shap_tree" and HAS_SHAP:
            try:
                # Pre-select representative sample indices to guarantee they are evaluated in SHAP
                y_pred_init = pipeline_or_model.predict(X_val_df)
                y_pred_arr_init = np.array(y_pred_init)

                if task_type.lower() == "classification":
                    init_samples = select_representative_samples_classification(y_val, y_pred_arr_init, max_samples=5)
                else:
                    init_samples = select_representative_samples_regression(y_val, y_pred_arr_init, max_samples=5)

                selected_row_indices = [idx for idx, _ in init_samples]
                bg_indices = list(range(len(X_trans)))
                eval_indices = list(dict.fromkeys(selected_row_indices + bg_indices))[:200]

                shap_res = explain_shap_tree(
                    model=model,
                    X_trans=X_trans,
                    feature_names=feature_names,
                    raw_sample_df=X_val_df,
                    max_samples=200,
                    eval_indices=eval_indices
                )
                global_recs = shap_res["global_importance"]
                warnings.extend(shap_res["warnings"])
                vals_array = shap_res["vals_array"]
                base_val = shap_res["base_value"]
                index_to_eval_pos = shap_res["index_to_eval_pos"]

                # Extract local SHAP contributions using exact index mapping
                def local_shap_extractor(orig_row_idx: int):
                    return get_local_shap_contributions(
                        vals_array=vals_array,
                        orig_row_idx=orig_row_idx,
                        feature_names=feature_names,
                        X_trans=X_trans,
                        raw_sample_df=X_val_df,
                        index_to_eval_pos=index_to_eval_pos
                    )

                local_recs = generate_local_explanations(
                    pipeline_or_model=pipeline_or_model,
                    X_val=X_val_df,
                    y_val=y_val,
                    X_trans=X_trans,
                    feature_names=feature_names,
                    task_type=task_type,
                    contribution_extractor_fn=local_shap_extractor,
                    base_value=base_val
                )
                executed_method = "shap_tree"
            except Exception as e:
                warnings.append(f"SHAP TreeExplainer failed ({str(e)}). Falling back to native/permutation.")
                primary_strategy = "native_tree" if hasattr(model, "feature_importances_") else "permutation_importance"

        if primary_strategy == "linear_coefficients":
            try:
                lin_res = explain_linear_coefficients(
                    model=model,
                    X_trans=X_trans,
                    feature_names=feature_names,
                    raw_sample_df=X_val_df
                )
                global_recs = lin_res["global_importance"]
                warnings.extend(lin_res["warnings"])
                raw_weights = lin_res["raw_weights"]
                intercept = lin_res["intercept"]

                # Extract local linear contributions using orig_row_idx
                def local_lin_extractor(orig_row_idx: int):
                    sample_row_trans = X_trans[orig_row_idx] if orig_row_idx < len(X_trans) else X_trans[0]
                    return get_local_linear_contributions(
                        raw_weights=raw_weights,
                        sample_row_trans=sample_row_trans,
                        feature_names=feature_names,
                        raw_sample_df=X_val_df,
                        orig_row_idx=orig_row_idx
                    )

                local_recs = generate_local_explanations(
                    pipeline_or_model=pipeline_or_model,
                    X_val=X_val_df,
                    y_val=y_val,
                    X_trans=X_trans,
                    feature_names=feature_names,
                    task_type=task_type,
                    contribution_extractor_fn=local_lin_extractor,
                    base_value=intercept
                )
                executed_method = "linear_coefficients"
            except Exception as e:
                warnings.append(f"Linear coefficient extraction failed ({str(e)}). Falling back to permutation.")
                primary_strategy = "permutation_importance"

        if primary_strategy == "native_tree":
            try:
                nat_res = explain_native_tree_importance(model=model, feature_names=feature_names)
                global_recs = nat_res["global_importance"]
                warnings.extend(nat_res["warnings"])

                # Native tree feature_importances_ is a global model-level measure (no per-sample attribution)
                local_recs = []
                warnings.append(
                    "Native feature_importances_ provides global model-level importance and does not provide native per-sample attribution."
                )
                executed_method = "native_tree"
            except Exception as e:
                warnings.append(f"Native tree importance failed ({str(e)}). Falling back to permutation.")
                primary_strategy = "permutation_importance"

        if primary_strategy == "permutation_importance":
            try:
                perm_feature_names = list(X_val_df.columns)
                perm_res = explain_permutation_importance(
                    pipeline_or_model=pipeline_or_model,
                    X_val=X_val_df,
                    y_val=y_val,
                    feature_names=perm_feature_names,
                    task_type=task_type,
                    n_repeats=n_repeats,
                    random_state=random_state
                )
                global_recs = perm_res["global_importance"]
                warnings.extend(perm_res["warnings"])
                scoring_metric = perm_res.get("scoring_metric", scoring_metric)

                # Permutation importance is a global model-level measure (no per-sample attribution)
                local_recs = []
                warnings.append(
                    "Permutation importance provides global model-level feature importance and does not provide native per-sample attribution."
                )
                executed_method = "permutation_importance"
            except Exception as e:
                warnings.append(f"Permutation importance failed ({str(e)}). Model is unsupported.")
                executed_method = "unsupported"
                local_recs = []

        # Format and rank global importances
        formatted_global = format_global_importance(global_recs)

        # Validate importances & check no NaN/Inf
        is_imp_valid, imp_errs = validate_importance_values(formatted_global)
        if not is_imp_valid:
            warnings.extend(imp_errs)

        end_time = time.perf_counter()
        elapsed_sec = round(end_time - start_time, 4)

        # Target leakage validation check
        target_name = None
        if y_val is not None:
            if hasattr(y_val, "name") and y_val.name:
                target_name = str(y_val.name)
            elif isinstance(y_val, pd.Series) and y_val.name:
                target_name = str(y_val.name)

        if target_name:
            leakage_detected = False
            for fi in formatted_global:
                if str(fi.feature) == target_name:
                    leakage_detected = True
                    break

            for loc in local_recs:
                if hasattr(loc, "contributions") and loc.contributions:
                    for contrib in loc.contributions:
                        if str(contrib.feature) == target_name:
                            leakage_detected = True
                            break
                elif isinstance(loc, dict) and "contributions" in loc:
                    for contrib in loc["contributions"]:
                        c_feat = contrib.get("feature") if isinstance(contrib, dict) else getattr(contrib, "feature", None)
                        if str(c_feat) == target_name:
                            leakage_detected = True
                            break

            if leakage_detected:
                error_msg = f"Target leakage detected: Target column '{target_name}' was found in the model's feature explanations."
                logger.error(error_msg)
                raise ValueError(error_msg)

        output = ExplanationOutput(
            dataset_id=dataset_id,
            pipeline_id=pipeline_id,
            model_name=actual_model_name,
            task_type=task_type,
            metric=metric,
            model_score=round(float(model_score), 4),
            method=executed_method,
            global_importance=formatted_global,
            local_explanations=local_recs,
            warnings=warnings,
            metadata={
                "scoring_metric": scoring_metric,
                "random_state": random_state,
                "n_repeats": n_repeats,
                "n_samples_explained": len(local_recs),
                "runtime_seconds": elapsed_sec,
                "has_preprocessor": preprocessor is not None
            }
        )

        # Verify output safety (no NaN/Inf)
        if not validate_no_nan_inf(output.model_dump()):
            warnings.append("Output contained non-finite values (NaN/Inf) which were handled.")

        return output
