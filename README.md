# GENESIS-AI: Dataset Intelligence Profile (DIP)-Guided AutoML

GENESIS-AI is a research-oriented tabular AutoML framework built around the hypothesis:
> *Can understanding a dataset before AutoML search reduce the search space and computational search cost while maintaining comparable predictive performance?*

This repository contains the baseline implementation of **Week 2: Dataset Intelligence Profile (DIP) v1.1**.

---

## Key Features (Week 2 — DIP v1.1)

- **Robust CSV Ingestion & Validation**: Safe file loading, format checking, empty file detection, and target column validation.
- **Comprehensive Feature & Type Profiling**:
  - Distinguishes continuous numerical, binary (0/1 numeric, boolean, 2-category string), multi-class categorical, boolean, and datetime features.
- **Separated Missingness Profiling**:
  - Feature missingness computed strictly over feature cells ($N_{\text{rows}} \times N_{\text{features}}$).
  - Target missingness computed separately on the target column.
- **Data Quality Profiling**: Missingness rate, duplicate row count & rate.
- **Statistical Profile Extraction**:
  - IQR-based outlier detection.
  - Mean & max feature skewness.
  - Pairwise-complete Pearson correlation matrix profiling & high-correlation pair counting ($|r| \ge 0.90$).
- **Target & Task Analysis**:
  - Transparent heuristic for classification vs. regression task type detection.
  - Imbalance ratio ($N_{\text{majority}} / N_{\text{minority}}$), minority percentage, class entropy for classification.
  - Target summary statistics for regression.
- **GENESIS DIP Complexity Score**: Transparent, configurable 0–10 engineering heuristic score combining missingness, outliers, skewness, imbalance, and feature-to-sample ratio.
- **Canonical SHA-256 Dataset Hashing**: OS-independent, deterministic dataset-content SHA-256 hash ensuring 100% research reproducibility.
- **FastAPI REST API**: Endpoints for dataset validation and DIP JSON generation (`/health`, `/dip`).
- **Comprehensive Pytest Suite**: Automated tests across 5 software-validation datasets with 100% pass rate.

---

## Software-Validation Datasets & Measured Results

| Dataset Filename | Task Type | Rows | Features | Feature Missing Rate | Outlier Rate | Imbalance Ratio | Complexity Score | Label | Canonical Dataset SHA-256 Hash |
|---|---|---|---|---|---|---|---|---|---|
| `01_numerical_classification.csv` | classification | 20 | 5 | 0.0000 | 0.0600 | 1.00 | **2.17** | **Low** | `cf953949b9e1838795bb00a1f9f634990f97e9a18f4d94c98ee434d1f31f01fe` |
| `02_categorical_heavy.csv` | classification | 12 | 5 | 0.0000 | 0.0000 | 1.00 | **1.67** | **Low** | `c87376d30a74467053c00316619d7fcfeb4e4b73764ebf17b627f79a8192d605` |
| `03_missing_values.csv` | classification | 10 | 4 | 0.2500 | 0.0000 | 1.00 | **3.44** | **Medium** | `13fb4dfed49021342270c2ee30fb2209660ebc5a030cbf44bd76cc58f21c69ea` |
| `04_imbalanced_classification.csv` | classification | 20 | 3 | 0.0000 | 0.1000 | 9.00 | **4.40** | **Medium** | `2567b9359765690a23ccdb83449a2cf0bdf3eeda263ef795754abc61fe70b62c` |
| `05_regression.csv` | regression | 15 | 4 | 0.0000 | 0.0000 | N/A | **1.24** | **Low** | `1ff50a62b3ac009c2bef01a30d422668d612286c56d2425278e35cf0313b87f1` |

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
│   ├── main.py                     # FastAPI application entrypoint
│   └── dataset/
│       ├── loader.py               # Safe CSV loader
│       ├── validator.py            # Dataset & target validator
│       ├── profiler.py             # Feature & statistical profilers
│       ├── complexity.py           # DIP Complexity Score calculator
│       └── dip.py                  # DIP builder and JSON serializer
├── tests/                          # Automated pytest suite
├── data/
│   └── test_datasets/              # 5 software-validation test CSVs
├── experiments/                    # Exported machine-readable DIP JSON outputs
├── docs/                           # Architectural and DIP v1.1 documentation
└── Memory.md                       # Project state tracking log
```

---

## Research Principles

1. **Non-destructive Profiling**: The DIP engine strictly observes dataset characteristics; it never mutates, drops, or imputes data silently.
2. **Determinism**: Identical data + target + config = identical DIP output.
3. **Transparency**: All metrics, formulas, and complexity score weights are explicit and configurable.
