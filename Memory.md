# GENESIS-AI Project Memory Log

## Current Status
- **Week 2 (Baseline)**: Completed & Frozen (Commit `4068902`, DIP v1.1)
- **Week 3 v1.1 (Research Hardening Pass)**: Completed, Verified, & Documented

---

## Week 3 v1.1 — Intelligent Pipeline Recommendation Engine (Hardening Pass)

### Implementation Status
Completed and fully verified with automated test suite (**69/69 passed**) and software validation experiments.

### Research Hardening Highlights (v1.1)
1. **Filtering Reduction Separation**: `filtering_reduction` ($1.0 - \text{after} / \text{before}$) is strictly separated from Top-K selection ratio (`top_k_selection_ratio`). Top-K selection parameters do not contaminate filtering metrics.
2. **ScoringWeights Validation**: Enforced strict weight sum validation ($\sum w_i == 1.0 \pm 1e-6$). Invalid weights, negative values, NaN, and Inf are rejected with explicit `ValueError`s without silent auto-normalization.
3. **HistGradientBoosting Metadata Alignment**: Corrected `handles_categorical_natively = False` for `HistGradientBoosting` entries to match their explicit `OrdinalEncoder` step.
4. **Heuristic Explanations & Rule Traceability**: Reasons use `RecommendationReason(rule_id=..., reason=...)` with structured rule codes and explicit heuristic wording.

### Files Created
1. `backend/recommendation/__init__.py`: Package entrypoint exposing public API.
2. `backend/recommendation/schemas.py`: Pydantic models (`PipelineMetadata`, `NormalizedDIPSignals`, `ScoringWeights`, `ThresholdConfig`, `RecommendationReason`, `RecommendationReport`, etc.).
3. `backend/recommendation/registry.py`: Candidate model & pipeline registry (10 scikit-learn candidates).
4. `backend/recommendation/normalizer.py`: DIP v1.1 signal normalizer & derived semantic flags.
5. `backend/recommendation/filters.py`: Stage 1 hard compatibility filters.
6. `backend/recommendation/rules.py`: Stage 2 deterministic recommendation rules & rule traceability.
7. `backend/recommendation/scorer.py`: Multi-criteria suitability scorer ($S \in [0.0, 1.0]$).
8. `backend/recommendation/ranker.py`: Top-K candidate ranker with alphabetic tie-breaking.
9. `backend/recommendation/engine.py`: Orchestrator `RecommendationEngine` class & `recommend_pipelines` helper.
10. `tests/test_recommendation_registry.py`: Registry unit tests.
11. `tests/test_recommendation_normalizer.py`: Normalizer unit tests.
12. `tests/test_recommendation_filters.py`: Hard filter unit tests.
13. `tests/test_recommendation_rules.py`: Soft rule engine unit tests.
14. `tests/test_recommendation_scorer.py`: Suitability scorer unit tests.
15. `tests/test_recommendation_ranker.py`: Top-K ranker unit tests.
16. `tests/test_recommendation_engine.py`: Recommendation engine integration & determinism tests.
17. `tests/test_recommendation_hardening.py`: Hardening regression tests for Fixes #1, #2, #3, and #4.
18. `experiments/generate_recommendation_results.py`: Validation script for 5 test datasets.
19. `experiments/week3_recommendation_results.json`: Measured output artifact.
20. `docs/Week3-Recommendation.md`: Technical documentation for Week 3 v1.1.

### Files Modified
1. `backend/main.py`: Integrated `POST /recommend` FastAPI endpoint.
2. `tests/test_api.py`: Added tests for `POST /recommend`.
3. `README.md`: Updated architecture, key features, API endpoints, test instructions, and dataset table.
4. `docs/Architecture.md`: Updated system flow and module breakdown.
5. `docs/Phases.md`: Marked Week 3 completed.

### Model & Pipeline Registry
- **Classification (5 Candidates)**:
  1. `classification_logistic_regression` (Logistic Regression)
  2. `classification_random_forest` (Random Forest Classifier)
  3. `classification_hist_gradient_boosting` (HistGradientBoosting Classifier)
  4. `classification_svc` (Support Vector Classifier)
  5. `classification_k_neighbors` (K-Neighbors Classifier)
- **Regression (5 Candidates)**:
  1. `regression_linear_regression` (Linear Regression)
  2. `regression_random_forest` (Random Forest Regressor)
  3. `regression_hist_gradient_boosting` (HistGradientBoosting Regressor)
  4. `regression_svr` (Support Vector Regressor)
  5. `regression_k_neighbors` (K-Neighbors Regressor)

### Scoring Weights
- `task`: 0.20
- `dataset_size`: 0.15
- `feature_type`: 0.20
- `missingness`: 0.10
- `imbalance`: 0.10
- `dimensionality`: 0.10
- `computational`: 0.15

### Pytest Execution Status
- Total tests: 69 (38 Week 2 tests + 31 Week 3 tests)
- Passed: 69
- Failed: 0
- Execution time: ~3.5s

### Software-Validation Dataset Results

| Dataset Filename | Task Type | Complexity Score | Candidates Before | Candidates After | Filtering Reduction | Configured Top-K | Recommended Count | Top-K Selection Ratio | Top Recommended Pipeline | Top Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|
| `01_numerical_classification.csv` | classification | 2.17 (Low) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `classification_logistic_regression` | 0.9775 |
| `02_categorical_heavy.csv` | classification | 1.67 (Low) | 10 | 4 | **60.0%** | 5 | 4 | 100.0% | `classification_hist_gradient_boosting` | 0.9450 |
| `03_missing_values.csv` | classification | 3.44 (Medium) | 10 | 4 | **60.0%** | 5 | 4 | 100.0% | `classification_hist_gradient_boosting` | 0.9350 |
| `04_imbalanced_classification.csv` | classification | 4.40 (Medium) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `classification_logistic_regression` | 0.9725 |
| `05_regression.csv` | regression | 1.24 (Low) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `regression_linear_regression` | 0.9775 |

### Research Methodology & Precautions
- **Filtering Reduction**: Filtering reduction (50–60%) measures search-space pruning caused by DIP compatibility rules prior to search.
- **No Hypothesis Claim**: Week 3 v1.1 provides a corrected, auditable recommendation mechanism for the Week 4/5 search-efficiency experiment without claiming hypothesis proof.
- **No Model Training / Tuning**: Zero model fitting or hyperparameter optimization was performed.

### Known Limitations
- Candidate registry relies on 10 scikit-learn baseline pipelines.
- Rules are engineering heuristics to be evaluated in Phase 8.

### Next Phase
- **Week 4/5 — Evolutionary Optimizer (DEAP)** *(Not started; awaiting explicit user instruction).*
