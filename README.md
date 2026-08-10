# GENESIS-AI: Dataset Intelligence Profile (DIP)-Guided AutoML

GENESIS-AI is a research-oriented tabular AutoML framework built around the hypothesis:
> *Can understanding a dataset before AutoML search reduce the search space and computational search cost while maintaining comparable predictive performance?*

This repository contains the implementation of **Week 2: Dataset Intelligence Profile (DIP) v1.1**, **Week 3: Intelligent Pipeline Recommendation Engine**, and **Week 4: Evolutionary Pipeline Optimization Engine**.

---

## Key Features

### Week 2 — Dataset Intelligence Profile (DIP) v1.1
- **Robust CSV Ingestion & Validation**: Safe file loading, format checking, empty file detection, and target column validation.
- **Comprehensive Feature & Type Profiling**:
  - Distinguishes continuous numerical, binary, multi-class categorical, boolean, and datetime features.
- **Separated Missingness Profiling**: Feature missingness vs target missingness computed separately.
- **Statistical Profile Extraction**: Outliers (IQR), skewness, Pearson correlation matrix.
- **GENESIS DIP Complexity Score**: Transparent 0–10 engineering heuristic score.
- **Canonical SHA-256 Dataset Hashing**: OS-independent, deterministic dataset-content SHA-256 hash.

### Week 3 v1.1 — Intelligent Pipeline Recommendation Engine (Research Hardening)
- **Central Model & Pipeline Registry**: 10 scikit-learn pipeline candidates (5 Classification, 5 Regression).
- **DIP Signal Normalization**: Extract semantic dataset flags without mutating raw DIP dictionary.
- **Stage 1 Compatibility Filtering**: Hard filters eliminating wrong task types or unsupported preprocessors.
- **Stage 2 Multi-Criteria Suitability Scoring**: Deterministic rule-based scoring ($S \in [0.0, 1.0]$) with strict weight validation (`sum == 1.0`).
- **Separated Filtering Reduction Metric**: Explicitly measures search-space pruning caused by compatibility filtering (`filtering_reduction`) separate from Top-K selection (`top_k_selection_ratio`).
- **Machine-Generated Heuristic Explanations with Rule Traceability**: Structured rule explanations with unique `rule_id` codes generated deterministically without LLMs.

### Week 4 — Evolutionary Pipeline Optimization Engine
- **Two Operational Modes**: `GENESIS` (restricted to Week 3 Top-K candidates) vs `BASELINE` (control group using all compatible pipelines).
- **Strict Data Isolation**: Train ($60\%$), Validation ($20\%$), and Test ($20\%$) splits. Test data is NEVER accessed during evolution and is evaluated strictly post-GA.
- **Evaluation Cache & Hard Budget Limit**: Cache tracking requests, unique evaluations, and cache hits while strictly enforcing `max_evaluations` budget limits.
- **Scikit-Learn Pipeline Construction**: Automatically builds and fits runnable `scikit-learn` Pipelines containing imputer, scaler/encoder, and model estimators.
- **Reproducibility**: Guaranteed deterministic execution under configured `random_state`.
- **FastAPI Endpoints**: `/health`, `/dip`, `/recommend`, `/optimize`.

---

## Software-Validation Datasets & Measured Results

| Dataset Filename | Task Type | Complexity Score | Candidates Before | Candidates After | Filtering Reduction | Configured Top-K | Recommended Count | Top-K Selection Ratio | Top Recommended Pipeline | Top Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|
| `01_numerical_classification.csv` | classification | 2.17 (Low) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `classification_logistic_regression` | 0.9775 |
| `02_categorical_heavy.csv` | classification | 1.67 (Low) | 10 | 4 | **60.0%** | 5 | 4 | 100.0% | `classification_hist_gradient_boosting` | 0.9450 |
| `03_missing_values.csv` | classification | 3.44 (Medium) | 10 | 4 | **60.0%** | 5 | 4 | 100.0% | `classification_hist_gradient_boosting` | 0.9350 |
| `04_imbalanced_classification.csv` | classification | 4.40 (Medium) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `classification_logistic_regression` | 0.9725 |
| `05_regression.csv` | regression | 1.24 (Low) | 10 | 5 | **50.0%** | 5 | 5 | 100.0% | `regression_linear_regression` | 0.9775 |

---

## Installation & Setup

1. **Clone repository**:
   ```bash
   git clone <repo-url>
   cd GENESIS-AI
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the API

Start the FastAPI application:
```bash
uvicorn backend.main:app --reload --port 8000
```

- Health Check: `GET http://localhost:8000/health`
- Generate DIP: `POST http://localhost:8000/dip` (Form data: `file` [CSV], `target_column` [string])
- Recommend Pipelines: `POST http://localhost:8000/recommend` (Form data: `file` [CSV], `target_column` [string], `top_k` [integer])
- Evolutionary Optimization: `POST http://localhost:8000/optimize` (Form data: `file` [CSV], `target_column` [string], `mode` [genesis/baseline], `population_size` [int], `generations` [int])

---

## Running Tests

Execute unit and integration tests:
```bash
python -m pytest tests/ -v
```

---

## Repository Structure

```text
GENESIS-AI/
├── backend/
│   ├── main.py                     # FastAPI application entrypoint (/dip, /recommend, /optimize)
│   ├── dataset/
│   │   ├── loader.py               # Safe CSV loader
│   │   ├── validator.py            # Dataset & target validator
│   │   ├── profiler.py             # Feature & statistical profilers
│   │   ├── complexity.py           # DIP Complexity Score calculator
│   │   └── dip.py                  # DIP builder and JSON serializer
│   ├── recommendation/
│   │   ├── schemas.py              # Pydantic models & data contracts
│   │   ├── registry.py             # Model & pipeline candidate registry
│   │   ├── normalizer.py           # DIP signal normalization & derived flags
│   │   ├── filters.py              # Stage 1 hard compatibility filters
│   │   ├── rules.py                # Stage 2 soft recommendation rules
│   │   ├── scorer.py               # Deterministic multi-criteria suitability scorer
│   │   ├── ranker.py               # Deterministic Top-K ranker
│   │   └── engine.py               # RecommendationEngine orchestrator
│   └── optimization/
│       ├── schemas.py              # Pydantic models & result contracts
│       ├── search_space.py         # Discrete hyperparameter search grids
│       ├── chromosome.py           # Chromosome individual structure
│       ├── cache.py                # Evaluation cache & metric counters
│       ├── evaluator.py            # Pipeline builder & fitness evaluator
│       ├── population.py           # Population initializer (GENESIS vs BASELINE)
│       ├── selection.py            # Tournament selection operator
│       ├── crossover.py            # Structural & hyperparameter crossover
│       ├── mutation.py             # Re-sampling mutation operator
│       ├── fitness.py              # Fitness manager & budget enforcer
│       ├── optimizer.py            # EvolutionaryOptimizer GA loop
│       └── modes.py                # GENESIS & BASELINE mode wrappers
├── tests/                          # Automated pytest suite (87 passing tests)
├── data/
│   └── test_datasets/              # 5 software-validation test CSVs
├── experiments/                    # Exported measured JSON output results
├── docs/                           # Documentation (DIP, Recommendation, Evolutionary Optimization)
└── Memory.md                       # Persistent project state tracking log
```

---

## Research Principles

1. **Non-destructive Profiling**: The engine strictly observes dataset characteristics; it never mutates, drops, or imputes data silently.
2. **Determinism**: Identical data + target + config = identical DIP & Recommendation output.
3. **Transparency**: All metrics, formulas, scoring weights, and rules are explicit, configurable, and explainable without LLMs.

