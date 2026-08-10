# GENESIS-AI: Dataset Intelligence Profile (DIP)-Guided AutoML

GENESIS-AI is a research-oriented tabular AutoML framework built around the hypothesis:
> *Can understanding a dataset before AutoML search reduce the search space and computational search cost while maintaining comparable predictive performance?*

This repository contains the implementation of:
- **Week 2: Dataset Intelligence Profile (DIP) v1.1**
- **Week 3: Intelligent Pipeline Recommendation Engine**
- **Week 4: Evolutionary Pipeline Optimization Engine**
- **Week 5: Explainability & Model Insight Engine**

---

## Key Features

### Week 2 — Dataset Intelligence Profile (DIP) v1.1
- **Robust CSV Ingestion & Validation**: Safe file loading, format checking, empty file detection, and target column validation.
- **Comprehensive Feature & Type Profiling**: Distinguishes continuous numerical, binary, multi-class categorical, boolean, and datetime features.
- **Separated Missingness Profiling**: Feature missingness vs target missingness computed separately.
- **Statistical Profile Extraction**: Outliers (IQR), skewness, Pearson correlation matrix.
- **GENESIS DIP Complexity Score**: Transparent 0–10 engineering heuristic score.
- **Canonical SHA-256 Dataset Hashing**: OS-independent, deterministic dataset-content SHA-256 hash.

### Week 3 v1.1 — Intelligent Pipeline Recommendation Engine
- **Central Model & Pipeline Registry**: 10 scikit-learn pipeline candidates (5 Classification, 5 Regression).
- **DIP Signal Normalization**: Extract semantic dataset flags without mutating raw DIP dictionary.
- **Stage 1 Compatibility Filtering**: Hard filters eliminating wrong task types or unsupported preprocessors.
- **Stage 2 Multi-Criteria Suitability Scoring**: Deterministic rule-based scoring ($S \in [0.0, 1.0]$) with strict weight validation (`sum == 1.0`).
- **Machine-Generated Heuristic Explanations with Rule Traceability**: Structured rule explanations with unique `rule_id` codes generated deterministically without LLMs.

### Week 4 — Evolutionary Pipeline Optimization Engine
- **Two Operational Modes**: `GENESIS` (restricted to Week 3 Top-K candidates) vs `BASELINE` (control group using all compatible pipelines).
- **Strict Data Isolation**: Train ($60\%$), Validation ($20\%$), and Test ($20\%$) splits. Test data is NEVER accessed during evolution and is evaluated strictly post-GA.
- **Evaluation Cache & Hard Budget Limit**: Cache tracking requests, unique evaluations, and cache hits while strictly enforcing `max_evaluations` budget limits.
- **Scikit-Learn Pipeline Construction**: Automatically builds and fits runnable `scikit-learn` Pipelines containing imputer, scaler/encoder, and model estimators.

### Week 5 — Explainability & Model Insight Engine
- **Post-Hoc Model Attribution**: Layered explanation strategy operating on fitted candidate pipelines without model retraining or weight mutation.
- **Strict Metric Scorer Enforcement**: Classification permutation importance evaluates macro F1 (`f1_macro`); regression permutation importance evaluates negative RMSE (`neg_root_mean_squared_error`).
- **Scientifically Valid Local Attribution Policy**:
  - `shap_tree` $\rightarrow$ global + local attributions (`RandomForest`, `HistGradientBoosting`).
  - `linear_coefficients` $\rightarrow$ global + local linear contributions ($w_i \cdot x_{trans, i}$) for linear models.
  - `permutation_importance` $\rightarrow$ global importances only (`local_explanations = []` + warning).
  - `native_tree` $\rightarrow$ global importances only (`local_explanations = []` + warning).
- **Representative Local Prediction Explanations**: Bounded explanations for up to 5 representative samples categorized by error type (TP, TN, FP, FN for classification; low/high/median residual for regression).
- **Row Alignment & Prediction Indexing**: Consistent mapping connecting prediction, actual target, and feature values to exact dataset row index `orig_row_idx`.
- **Schema & Validation Safety**: Strict Pydantic v2 data models, rank validation, non-finite (NaN/Inf) output protection, and test set isolation.

---

## Software-Validation Datasets & Measured Results

| Dataset Filename | Task Type | Best Week 4 Pipeline | Score | Explanation Method | Top Feature | Top Imp | Local Samples |
|---|:---:|---|:---:|:---:|---|:---:|:---:|
| `01_numerical_classification.csv` | classification | `classification_svc` | 0.3333 | `permutation_importance` | `chol` | 1.0000 | 0 |
| `02_categorical_heavy.csv` | classification | `classification_random_forest` | 1.0000 | `shap_tree` | `department_HR` | 0.2515 | 3 |
| `03_missing_values.csv` | classification | `classification_svc` | 0.3333 | `permutation_importance` | `category_x` | 0.0000 | 0 |
| `04_imbalanced_classification.csv` | classification | `classification_svc` | 1.0000 | `permutation_importance` | `v2` | 0.0000 | 0 |
| `05_regression.csv` | regression | `regression_linear_regression` | -8302.48 | `linear_coefficients` | `square_feet` | 0.7203 | 3 |

---

## Installation & Setup

### Requirements
- Python 3.9+
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `pydantic>=2.0`, `pytest`, `shap>=0.41.0`

```bash
# Clone the repository
git clone https://github.com/Rahulpurbey0123/GENESIS-AI.git
cd GENESIS-AI

# Install dependencies
pip install -r requirements.txt

# Run full test suite (132 passed)
python -m pytest tests/ -q
```

---

## Running Development Experiments

```bash
# Run Week 2 DIP Extraction
python -m experiments.run_dip_v1_1_experiment

# Run Week 3 Recommendation Experiment
python -m experiments.generate_recommendation_results

# Run Week 4 Evolutionary Optimization Experiment
python -m experiments.generate_optimization_results

# Run Week 5 Explainability Experiment
python -m experiments.generate_explainability_results
```

---

## Verification & Test Suite
The codebase is covered by **132 unit and integration tests** passing with 0 failures:
- Week 2 Tests: 38 passing tests
- Week 3 Tests: 31 passing tests
- Week 4 Tests: 30 passing tests
- Week 5 Tests: 33 passing tests (including methodological corrections & row-alignment verification)
