# GENESIS-AI Week 4.1 — Evolutionary Pipeline Optimization Engine (Research Hardening & Correction Pass)

## Executive Overview

Week 4.1 implements the research-hardened **Evolutionary Pipeline Optimization Engine** (`backend/optimization/`), introducing model training, fitness evaluation, and Genetic Algorithm (GA) optimization to search for the best pipeline and hyperparameters.

The flow of GENESIS-AI is:

```
Dataset ──► DIP v1.1 ──► Week 3 Recommendation Engine ──► Candidate Space ──► Evolutionary Optimization ──► Best Pipeline + Hyperparameters
```

Week 4.1 fixes research-critical issues around candidate space reduction, configurable Top-K selection, model-family mutation, and worst-case fitness failure handling.

---

## 1. Key v4.1 Hardening & Research Correction Enhancements

1. **True Candidate Space Reduction & Configurable Top-K**:
   - `top_k` parameter is fully configurable (`top_k = 1, 2, 3, 4, 5...`).
   - `GENESIS` mode restricts initial candidate search space to Week 3 Top-K recommendations.
   - `BASELINE` mode uses all compatible candidates as a control condition.
   - Explicit result metadata added: `candidate_count_before`, `candidate_count_after`, `candidate_space_reduction`, `candidate_pipeline_ids`.

2. **Controlled Model-Family Mutation**:
   - Introduced `pipeline_mutation_rate` (default `0.10`) separate from hyperparameter `mutation_rate` (default `0.10`).
   - `pipeline_id` is an evolvable gene. When pipeline mutation fires, a new `pipeline_id` is chosen strictly from the mode's allowed candidate pool (`candidate_pipeline_ids`).
   - Hyperparameters for the new model family are **regenerated from scratch** from the new model's discrete search grid (never carrying over incompatible parameters from the old model family).

3. **Worst-Case Fitness Handling (`float("-inf")`)**:
   - Replaced fixed fallbacks (like `-999.0`) with `float("-inf")` for both classification and regression failures.
   - Guarantees that failed candidate models can never beat any valid finite model score.

4. **Linear Regression Search Space Specification**:
   - Explicitly documents Linear Regression discrete search grid `fit_intercept: [True, False]` for hyperparameter optimization consistency.

---

## 2. Two Operational Modes

### GENESIS Mode (Experimental Group)
- Restricts initial candidate pipeline pool to Top-K recommendations output by the Week 3 Recommendation Engine (default `top_k = 2`).
- Evaluates whether restricting search space based on dataset intelligence reduces search time and evaluation count while achieving comparable predictive accuracy.

### BASELINE Mode (Control Group)
- Uses all compatible candidate pipelines from Stage 1 filtering as the search space.
- Serves as the control benchmark for future research experiments (Phase 8).

Both modes operate under identical conditions: dataset split, random seed (`42`), fitness evaluation metric, GA parameters, hyperparameter search grids, and evaluation budget (`max_evaluations = 200`).

---

## 3. System Architecture & Module Structure

The package is located at `backend/optimization/`:

- [`schemas.py`](file:///d:/GENESIS-AI/backend/optimization/schemas.py): Pydantic data contracts (`OptimizationConfig`, `ChromosomeDict`, `GenerationHistory`, `OptimizationResult`).
- [`search_space.py`](file:///d:/GENESIS-AI/backend/optimization/search_space.py): Discrete hyperparameter search spaces for all 10 registry pipelines.
- [`chromosome.py`](file:///d:/GENESIS-AI/backend/optimization/chromosome.py): Individual chromosome structure (`pipeline_id` + `hyperparameters`), hashing, serialization, and validation.
- [`cache.py`](file:///d:/GENESIS-AI/backend/optimization/cache.py): Evaluation cache tracking requests, unique evaluations, and cache hits.
- [`evaluator.py`](file:///d:/GENESIS-AI/backend/optimization/evaluator.py): Constructs `sklearn` pipelines, fits models, computes validation metrics, and handles exceptions gracefully with `float("-inf")` fallbacks.
- [`population.py`](file:///d:/GENESIS-AI/backend/optimization/population.py): Initial population generation for GENESIS and BASELINE modes.
- [`selection.py`](file:///d:/GENESIS-AI/backend/optimization/selection.py): Deterministic tournament selection (`tournament_size = 3`).
- [`crossover.py`](file:///d:/GENESIS-AI/backend/optimization/crossover.py): Model-aware crossover operator (`crossover_rate = 0.80`).
- [`mutation.py`](file:///d:/GENESIS-AI/backend/optimization/mutation.py): Model-family mutation (`pipeline_mutation_rate = 0.10`) and hyperparameter grid mutation (`mutation_rate = 0.10`).
- [`fitness.py`](file:///d:/GENESIS-AI/backend/optimization/fitness.py): High-level fitness manager enforcing hard evaluation budget limits (`max_evaluations = 200`).
- [`optimizer.py`](file:///d:/GENESIS-AI/backend/optimization/optimizer.py): Core GA orchestrator (`EvolutionaryOptimizer`).
- [`modes.py`](file:///d:/GENESIS-AI/backend/optimization/modes.py): Convenience helper wrappers `run_genesis_mode` and `run_baseline_mode`.

---

## 4. Chromosome Design & Search Spaces

Each individual chromosome is represented as:

```json
{
  "pipeline_id": "classification_random_forest",
  "hyperparameters": {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt"
  }
}
```

Discrete search space grids are defined for all 10 candidate pipelines in the Week 3 registry:
- **Random Forest**: `n_estimators` [100, 200, 300, 500], `max_depth` [None, 5, 10, 20, 30], `min_samples_split` [2, 5, 10], `min_samples_leaf` [1, 2, 4], `max_features` ["sqrt", "log2"].
- **Logistic Regression**: `C` [0.01, 0.1, 1.0, 10.0, 100.0], `solver` ["lbfgs"], `max_iter` [500, 1000, 2000].
- **SVC / SVR**: `C` [0.1, 1.0, 10.0, 100.0], `kernel` ["linear", "rbf"], `gamma` ["scale", "auto"], (SVR) `epsilon` [0.01, 0.1, 0.2].
- **KNN**: `n_neighbors` [3, 5, 7, 11, 15], `weights` ["uniform", "distance"], `p` [1, 2].
- **HistGradientBoosting**: `learning_rate` [0.01, 0.05, 0.1, 0.2], `max_iter` [100, 200, 300], `max_leaf_nodes` [15, 31, 63], `max_depth` [None, 5, 10], `l2_regularization` [0.0, 0.1, 1.0].
- **Linear Regression**: `fit_intercept` [True, False].

---

## 5. Strict Train / Validation / Test Separation

Data splitting is strictly enforced:
- **Train Split ($60\%$)**: Fits models during GA evolution.
- **Validation Split ($20\%$)**: Evaluates fitness during GA evolution.
- **Test Split ($20\%$)**: **Strictly isolated**. Test data is NEVER seen or evaluated during GA evolution. The final selected best pipeline is evaluated ONCE post-GA on the test set to report unbiased test metrics (`f1`, `accuracy`, `rmse`, `mae`, `r2`).

---

## 6. Software Validation Dataset Results (`top_k = 2`)

Measured output results on 5 software test datasets (`experiments/week4_optimization_results.json`):

| Dataset Filename | Task Type | Mode | Before/After | Candidate Reduction | Unique Evals | Cache Hits | Best Pipeline ID | Best Val Fitness | Test F1 / RMSE | Runtime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|:---:|:---:|
| `01_numerical_classification.csv` | classification | GENESIS | 5 -> 2 | **60.0%** | 28 | 172 | `classification_svc` | 0.3333 | F1: 0.7333 | 0.58s |
| `01_numerical_classification.csv` | classification | BASELINE | 5 -> 5 | **0.0%** | 49 | 151 | `classification_k_neighbors` | 0.7333 | F1: 0.7333 | 4.90s |
| `02_categorical_heavy.csv` | classification | GENESIS | 4 -> 2 | **50.0%** | 78 | 122 | `classification_random_forest` | 1.0000 | F1: 0.4000 | 19.22s |
| `02_categorical_heavy.csv` | classification | BASELINE | 4 -> 4 | **0.0%** | 42 | 158 | `classification_svc` | 1.0000 | F1: 1.0000 | 4.65s |
| `03_missing_values.csv` | classification | GENESIS | 4 -> 2 | **50.0%** | 57 | 143 | `classification_svc` | 0.3333 | F1: 0.3333 | 3.54s |
| `03_missing_values.csv` | classification | BASELINE | 4 -> 4 | **0.0%** | 71 | 129 | `classification_svc` | 0.3333 | F1: 0.3333 | 11.10s |
| `04_imbalanced_classification.csv` | classification | GENESIS | 5 -> 2 | **60.0%** | 28 | 172 | `classification_svc` | 1.0000 | F1: 1.0000 | 0.37s |
| `04_imbalanced_classification.csv` | classification | BASELINE | 5 -> 5 | **0.0%** | 69 | 131 | `classification_svc` | 1.0000 | F1: 1.0000 | 7.58s |
| `05_regression.csv` | regression | GENESIS | 5 -> 2 | **60.0%** | 26 | 174 | `regression_linear_regression` | -8302.48 | RMSE: 10131.51 | 0.27s |
| `05_regression.csv` | regression | BASELINE | 5 -> 5 | **0.0%** | 40 | 160 | `regression_linear_regression` | -8302.48 | RMSE: 10131.51 | 7.51s |

---

## 7. REST API Endpoint

FastAPI endpoint added to `backend/main.py`:

```http
POST /optimize
Content-Type: multipart/form-data

file: <CSV file>
target_column: <string>
mode: genesis | baseline
top_k: 2
population_size: 20
generations: 10
max_evaluations: 200
mutation_rate: 0.10
pipeline_mutation_rate: 0.10
random_state: 42
```
