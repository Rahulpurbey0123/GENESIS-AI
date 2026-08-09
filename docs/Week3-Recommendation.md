# GENESIS-AI Week 3 v1.1 — Intelligent Pipeline Recommendation Engine (Research Hardening)

## Executive Overview

Week 3 v1.1 implements the research-hardened **Intelligent Pipeline Recommendation Engine**, the deterministic decision-making layer between the **Dataset Intelligence Profile (DIP v1.1)** and future **Week 4/5 Evolutionary Optimization**.

The central research question guiding GENESIS-AI is:
> *"Can dataset understanding reduce the AutoML search space while maintaining comparable predictive performance?"*

Week 3 v1.1 builds the deterministic, transparent mechanism required to test this hypothesis by pruning incompatible candidate pipelines and ranking compatible pipelines using explicit dataset intelligence rules without model fitting, hyperparameter tuning, or LLM inference.

---

## 1. Key v1.1 Research Hardening Enhancements

1. **Separation of Filtering Reduction & Top-K Selection Ratio**:
   - `filtering_reduction`: Measures strict candidate search-space pruning resulting from hard DIP compatibility filtering ($1.0 - \text{candidate\_count\_after\_filtering} / \text{candidate\_count\_before}$).
   - `top_k_selection_ratio`: Separately reports Top-K post-filtering selection ($\text{recommended\_count} / \text{candidate\_count\_after\_filtering}$).
   - Top-K selection parameters do NOT contaminate filtering reduction metrics.

2. **ScoringWeights Strict Sum Validation**:
   - `ScoringWeights` enforces that all sub-score weights sum to $1.0$ within a floating-point tolerance ($\le 1e-6$).
   - Rejects invalid configurations, negative weights, NaN, and infinity with explicit validation errors without silent auto-normalization.

3. **HistGradientBoosting Registry Metadata Alignment**:
   - Accurately sets `handles_categorical_natively = False` for `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` registry entries, reflecting their actual pipeline definition which includes an `OrdinalEncoder` preprocessing step.

4. **Explicit Heuristic Explanations & Rule Traceability**:
   - Generated reasons state what rule fired under configured heuristics rather than claiming guaranteed optimal model performance.
   - Introduced `RecommendationReason` schema providing structured `rule_id` traceability (e.g. `RULE_SIZE_SMALL_HIGH_SUITABILITY`, `RULE_IMBALANCE_CLASS_WEIGHT_BOOST`, `RULE_DIM_HIGH_LINEAR_REGULARIZATION`).

---

## 2. High-Level Architecture

```
                          Uploaded Dataset
                                 │
                                 ▼
                     Dataset Intelligence Profile (DIP v1.1)
                                 │
                                 ▼
                      DIP Signal Normalizer
                                 │
                                 ▼
                     Candidate Pipeline Registry (10 Pipelines)
                                 │
                                 ▼
                    Stage 1: Hard Compatibility Filters
                     (Task & Preprocessing Compatibility)
                                 │
                                 ▼
                    Stage 2: Deterministic Rule Engine
                    (Multi-Criteria Suitability Scoring)
                                 │
                                 ▼
                     Deterministic Candidate Ranker
                     (Descending Score + Tie-Breaking)
                                 │
                                 ▼
                         Top-K Recommended Pipelines
                                 │
                                 ▼
                    Recommendation Report v1.1
               (filtering_reduction & top_k_selection_ratio)
                                 │
                                 ▼
                   Future Week 4/5 Evolutionary Optimizer
```

---

## 3. Central Candidate Pipeline Registry

The initial registry defines 10 candidate machine learning pipelines (5 for Classification, 5 for Regression) built exclusively using `scikit-learn`:

### Classification Pipelines

| Pipeline ID | Model Name | Model Family | Preprocessing Steps | Scaling | Class Weight Support | Native Categorical | Computational Cost |
|---|---|---|---|:---:|:---:|:---:|:---:|
| `classification_logistic_regression` | LogisticRegression | Linear | Imputer → Scaler → Model | Yes | Yes | No | Low |
| `classification_random_forest` | RandomForestClassifier | Tree Ensemble | Imputer → OneHotEncoder → Model | No | Yes | No | Medium |
| `classification_hist_gradient_boosting` | HistGradientBoostingClassifier | Tree Ensemble | Imputer → OrdinalEncoder → Model | No | Yes | No | Low |
| `classification_svc` | SVC | SVM | Imputer → OneHotEncoder → Scaler → Model | Yes | Yes | No | High |
| `classification_k_neighbors` | KNeighborsClassifier | KNN | Imputer → OneHotEncoder → Scaler → Model | Yes | No | No | Medium |

### Regression Pipelines

| Pipeline ID | Model Name | Model Family | Preprocessing Steps | Scaling | Class Weight Support | Native Categorical | Computational Cost |
|---|---|---|---|:---:|:---:|:---:|:---:|
| `regression_linear_regression` | LinearRegression | Linear | Imputer → OneHotEncoder → Scaler → Model | Yes | No | No | Low |
| `regression_random_forest` | RandomForestRegressor | Tree Ensemble | Imputer → OneHotEncoder → Model | No | No | No | Medium |
| `regression_hist_gradient_boosting` | HistGradientBoostingRegressor | Tree Ensemble | Imputer → OrdinalEncoder → Model | No | No | No | Low |
| `regression_svr` | SVR | SVM | Imputer → OneHotEncoder → Scaler → Model | Yes | No | No | High |
| `regression_k_neighbors` | KNeighborsRegressor | KNN | Imputer → OneHotEncoder → Scaler → Model | Yes | No | No | Medium |

---

## 4. DIP Signal Normalization & Scoring Formula

Each compatible candidate pipeline is evaluated across 7 weighted component sub-scores ($S_i \in [0.0, 1.0]$):

$$\text{Suitability Score} = \sum_{i=1}^{7} w_i S_i$$

### Scoring Weight Allocation ($w_i$)

| Sub-Score Component | Symbol | Configured Weight ($w_i$) | Validation Constraint |
|---|:---:|:---:|---|
| **Task Compatibility** | $S_{\text{task}}$ | 0.20 | Must sum to $1.0$ ($\le 1e-6$ tolerance) |
| **Dataset Size** | $S_{\text{size}}$ | 0.15 | Must sum to $1.0$ ($\le 1e-6$ tolerance) |
| **Feature Type** | $S_{\text{feat}}$ | 0.20 | Must sum to $1.0$ ($\le 1e-6$ tolerance) |
| **Missingness** | $S_{\text{miss}}$ | 0.10 | Must sum to $1.0$ ($\le 1e-6$ tolerance) |
| **Class Imbalance** | $S_{\text{imb}}$ | 0.10 | Must sum to $1.0$ ($\le 1e-6$ tolerance) |
| **Dimensionality** | $S_{\text{dim}}$ | 0.10 | Must sum to $1.0$ ($\le 1e-6$ tolerance) |
| **Computational Footprint** | $S_{\text{comp}}$ | 0.15 | Must sum to $1.0$ ($\le 1e-6$ tolerance) |

---

## 5. Software Validation Report (5 Baseline Datasets)

Measured execution results on the 5 GENESIS-AI software validation datasets:

| Dataset Filename | Task Type | DIP Complexity | Candidates Before | Candidates After | Filtering Reduction | Configured Top-K | Recommended Count | Top-K Selection Ratio | Top Recommended Pipeline | Top Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|
| `01_numerical_classification.csv` | classification | 2.17 (Low) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `classification_logistic_regression` | 0.9775 |
| `02_categorical_heavy.csv` | classification | 1.67 (Low) | 10 | 4 | **60.0%** | 5 | 4 | 100.0% | `classification_hist_gradient_boosting` | 0.9450 |
| `03_missing_values.csv` | classification | 3.44 (Medium) | 10 | 4 | **60.0%** | 5 | 4 | 100.0% | `classification_hist_gradient_boosting` | 0.9350 |
| `04_imbalanced_classification.csv` | classification | 4.40 (Medium) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `classification_logistic_regression` | 0.9725 |
| `05_regression.csv` | regression | 1.24 (Low) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `regression_linear_regression` | 0.9775 |

---

## 6. REST API Endpoint

FastAPI endpoint added to `backend/main.py`:

```http
POST /recommend
Content-Type: multipart/form-data

file: <CSV file>
target_column: <string>
top_k: <integer, default 5>
```

---

## 7. Research Leakage & Scope Precautions

1. **Software Validation vs Research Evaluation**: The 5 test datasets demonstrate engine execution, rule execution, and filter metrics. Research benchmarking will evaluate predictive performance against unrestricted search in Phase 8.
2. **No Model Fitting / Benchmarking**: Week 3 v1.1 does not fit models or tune hyperparameters.
3. **Foundation for Week 4/5**: Week 3 v1.1 produces candidate search spaces for future evolutionary search without implementing Genetic Algorithms or DEAP.
