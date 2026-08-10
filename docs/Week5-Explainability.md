# GENESIS-AI — Week 5: Explainability & Model Insight Engine

## 1. Objective
The **Explainability & Model Insight Engine** adds a deterministic, post-hoc model attribution layer to GENESIS-AI following Week 4 evolutionary optimization. After GENESIS selects and fits the best candidate machine learning pipeline, the Explainability Engine provides structured, reproducible evidence explaining:
- What candidate model/pipeline was selected.
- Which features are globally most important to the fitted model.
- Why the model produced specific local predictions for representative samples (for supported local methods).
- Which explanation strategy was selected and executed (`shap_tree`, `linear_coefficients`, `native_tree`, `permutation_importance`, or `unsupported`).
- What technical warnings and research limitations apply.

> [!IMPORTANT]
> The Explainability Engine is strictly **post-hoc**. It does **not** retrain fitted estimators, alter candidate search space rankings, tune hyperparameters, modify GA fitness scores, or introduce data leakage.

---

## 2. Core Architecture

```text
                               GENESIS-AI Pipeline Sequence
                               ============================

  [Dataset] ---> [DIP Engine (Week 2)] ---> [Recommendation Engine (Week 3)]
                                                   |
                                                   v
  [Best Pipeline] <--- [Evolutionary Optimization (Week 4)] <--- [Candidate Space]
         |
         v
  [Explainability Engine (Week 5)] ---> [Structured Explanation JSON] ---> [Future Week 6 LLM]
```

### Module Structure (`backend/explainability/`)
- `__init__.py`: Package export interface (`ExplainabilityEngine`, `ExplanationOutput`, schemas, validators).
- `schemas.py`: Pydantic v2 schemas (`FeatureImportanceRecord`, `LocalExplanationRecord`, `FeatureContribution`, `ExplanationOutput`).
- `registry.py`: Model explanation strategy registry mapping scikit-learn estimators to explanation strategies.
- `validators.py`: Strict validation for fitted models, feature names, finite importances, ranks, local indices, and non-finite (NaN/Inf) output protection.
- `shap_explainer.py`: SHAP `TreeExplainer` module for tree-based ensemble models (`RandomForest`, `HistGradientBoosting`).
- `native_importance.py`: Extractor for native `feature_importances_` (Gini/MDI) and linear coefficients (`coef_` weights and direction).
- `permutation.py`: Permutation feature importance fallback module evaluated on held-out validation data splits ($X_{val}, y_{val}$).
- `global.py`: Formatter and ranker for global feature importances.
- `local.py`: Representative sample selector (classification error types and regression residuals) and local explanation generator.
- `engine.py`: Central coordinator orchestrating model validation, strategy selection, global ranking, local explanations, and output validation.

---

## 3. Explanation Capability Policy

Local prediction explanations are produced **only** when the explanation method provides mathematically valid per-sample attributions.

| Strategy | Global Importance | Local Explanations | Scoring / Attribution Basis |
|---|:---:|:---:|---|
| **`shap_tree`** | Yes | Yes | SHAP values ($\phi_{i,j}$) & expected base value. |
| **`linear_coefficients`** | Yes | Yes | Transformed linear contribution ($w_j \cdot x_{trans, j}$). |
| **`permutation_importance`** | Yes | **No** (`[]`) | Held-out validation score drop (`f1_macro` / `neg_root_mean_squared_error`). |
| **`native_tree`** | Yes | **No** (`[]`) | Normalized Gini / MDI tree importance (`feature_importances_`). |
| **`unsupported`** | No | **No** (`[]`) | Structured error/warning response. |

> [!WARNING]
> Neither `permutation_importance` nor native `feature_importances_` provides valid per-sample attribution. The system does **not** construct artificial local contributions (such as $\text{importance}_j \cdot x_j$). For these global-only methods, `local_explanations` returns an empty list (`[]`) accompanied by an explicit warning.

---

## 4. Global Feature Importance Strategy

Global feature importance quantifies overall feature significance across the dataset:
1. **SHAP TreeExplainer**: Mean absolute SHAP value across samples ($\frac{1}{N} \sum_i |\phi_{i,j}|$), normalized to sum to 1.0.
2. **Linear Coefficients**: Normalized absolute coefficient magnitude ($\frac{|w_j|}{\sum_k |w_k|}$), with explicit direction indicator ($+1$ for positive, $-1$ for negative, `None` for multi-class).
3. **Native Tree Importance**: Normalized Gini / Mean Decrease in Impurity (MDI) importances.
4. **Permutation Importance**: Mean validation performance drop when feature values are randomly shuffled ($X_{val}, y_{val}$).
   - **Classification**: Uses macro F1 (`f1_macro`).
   - **Regression**: Uses negative RMSE (`neg_root_mean_squared_error`).
     *Note: Negative RMSE is used internally because scikit-learn permutation importance assumes higher scores are better, while research reporting interprets lower RMSE as superior predictive performance.*

Every global feature record includes:
- `feature`: Feature column name.
- `importance`: Normalized score $\in [0.0, 1.0]$.
- `rank`: 1-indexed rank order (1 is most important).
- `direction`: Optional direction indicator (+1, -1, or None).
- `mean_importance` & `std_importance`: Available for permutation importance.

---

## 5. Local Prediction Explanations

Local explanations explain *why* the model produced a specific output for up to 5 representative samples (for `shap_tree` and `linear_coefficients` methods):

### Representative Sample Selection Rules
- **Classification**:
  1. `correct_positive` (True Positive: $y=1, \hat{y}=1$)
  2. `correct_negative` (True Negative: $y=0, \hat{y}=0$)
  3. `false_positive` (False Positive: $y=0, \hat{y}=1$)
  4. `false_negative` (False Negative: $y=1, \hat{y}=0$)
  5. `representative_sample` (First available unused sample after higher-priority error-category samples)
- **Regression**:
  1. `low_residual` (Smallest absolute error $|y - \hat{y}|$)
  2. `high_residual` (Largest absolute error $|y - \hat{y}|$)
  3. `median_residual` (Median absolute error)
  4. `representative_sample_1` (25th percentile error sample)
  5. `representative_sample_2` (75th percentile error sample)

Row Alignment: `sample_index`, `prediction` ($\hat{y}_{orig\_row\_idx}$), `actual_value` ($y_{orig\_row\_idx}$), and `contributions` consistently reference the exact original row index `orig_row_idx`.

---

## 6. Validation & Leakage Prevention Safety

- **Post-Hoc Execution**: The Explainability Engine accepts pre-fitted pipelines. It never calls `.fit()` or alters fitted estimator weights.
- **No Hyperparameter Alteration**: Hyperparameters of chromosomes remain untouched.
- **Test Set Isolation**: Permutation importance evaluates on held-out validation data (`X_val`, `y_val`). The final test set (`X_test`) is **never** used for explanation tuning.
- **Zero NaN / Inf Output Guarantee**: Verified by `validate_no_nan_inf` recursion.
- **No Optimization Feedback**: Explanation results do not influence GA fitness, candidate space filtering, or pipeline ranking.

---

## 7. Controlled Development Experiment Results

The Week 5 controlled experiment (`experiments/generate_explainability_results.py`) evaluated the best Week 4 pipelines across all 5 development datasets:

| Dataset Filename | Task Type | Best Pipeline | Score | Method Used | Top Feature | Top Imp | Local Samples | Runtime |
|---|:---:|---|:---:|:---:|---|:---:|:---:|:---:|
| `01_numerical_classification.csv` | Classification | `classification_svc` | 0.3333 | `permutation_importance` | `chol` | 1.0000 | 0 | 0.19s |
| `02_categorical_heavy.csv` | Classification | `classification_random_forest` | 1.0000 | `shap_tree` | `department_HR` | 0.2515 | 3 | 0.03s |
| `03_missing_values.csv` | Classification | `classification_svc` | 0.3333 | `permutation_importance` | `category_x` | 0.0000 | 0 | 0.34s |
| `04_imbalanced_classification.csv` | Classification | `classification_svc` | 1.0000 | `permutation_importance` | `v2` | 0.0000 | 0 | 0.11s |
| `05_regression.csv` | Regression | `regression_linear_regression` | -8302.48 | `linear_coefficients` | `square_feet` | 0.7203 | 3 | 0.01s |

Results are exported to [experiments/week5_explainability_results.json](file:///d:/GENESIS-AI/experiments/week5_explainability_results.json).

---

## 8. Test Suite Verification

- **Total Test Baseline**: **132 passed** (99 baseline tests + 33 Week 5 explainability tests).
- **Test Command**: `python -m pytest tests/ -q`
- **Execution Time**: ~36.6s.

---

## 9. Research Limitations & Cautionary Notes

1. **Attribution, Not Causation**: Feature importances (SHAP, linear coefficients, permutation importance) measure statistical feature attribution and model dependency, **not** causal effect.
2. **Model Dependency**: Permutation importance and SHAP explanations depend on the specific trained model architecture and hyperparameter configuration.
3. **Correlated Features**: Correlated input features can split or redistribute feature importances across correlated columns.
4. **Method Disagreement**: Different explanation methods may yield different feature rankings due to underlying mathematical formulations.
5. **Raw Negative Permutation Importance**: Negative raw permutation importance indicates that shuffling the feature did not reduce the measured validation score and may have slightly improved it under that specific sample split.

---

## 10. Week 6 LLM Interface Readiness

The structured JSON output produced by `ExplanationOutput.model_dump()` is fully formatted for seamless consumption by a future Week 6 LLM narrative generation layer.
