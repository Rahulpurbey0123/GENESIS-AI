# Week 2 Specification — Dataset Intelligence Profile (DIP) v1.1

## 1. Overview

The **Dataset Intelligence Profile (DIP) v1.1** is a central research artifact in **GENESIS-AI**. It provides a deterministic, lightweight, non-destructive, model-independent summary of a tabular dataset prior to AutoML search space generation.

### Key Architectural Principles
1. **Determinism**: For an identical dataset, target column, and configuration, DIP v1.1 yields identical profile metrics and dataset SHA-256 hash.
2. **Non-destructive Profiling**: DIP measures dataset properties without modifying, dropping, or imputing raw data.
3. **Separated Missingness**: Explicitly separates feature-level missingness from target-level missingness to guide downstream AutoML recommendation accurately.
4. **Binary Feature Recognition**: Distinguishes continuous numerical features from binary features (whether 0/1 numeric, boolean, or 2-category string) to support algorithm selection.
5. **Transparent Engineering Heuristic**: Complexity score is explicitly defined as an interpretable engineering heuristic ($0.0 - 10.0$) rather than a claim of universal complexity law.
6. **Software Validation Datasets**: Benchmark experiments are conducted on 5 controlled software-validation datasets to ensure software correctness and edge-case coverage.

---

## 2. Complete DIP JSON Schema (v1.1)

```json
{
  "dip_version": "1.1",
  "dataset_hash": "cf953949b9e1838795bb00a1f9f634990f97e9a18f4d94c98ee434d1f31f01fe",
  "dataset": {
    "name": "01_numerical_classification.csv",
    "rows": 20,
    "columns": 6,
    "feature_count": 5,
    "memory_bytes": 1092
  },
  "schema": {
    "numeric_features": 5,
    "categorical_features": 0,
    "binary_features": 0,
    "boolean_features": 0,
    "datetime_features": 0,
    "numeric_ratio": 1.0,
    "categorical_ratio": 0.0,
    "binary_ratio": 0.0
  },
  "quality": {
    "total_missing": 0,
    "missing_rate": 0.0,
    "columns_with_missing": 0,
    "max_column_missing_rate": 0.0,
    "feature_missingness": {
      "total_missing": 0,
      "missing_rate": 0.0,
      "columns_with_missing": 0,
      "max_column_missing_rate": 0.0,
      "per_feature_missing_rates": {
        "age": 0.0,
        "trestbps": 0.0,
        "chol": 0.0,
        "thalach": 0.0,
        "oldpeak": 0.0
      }
    },
    "target_missingness": {
      "missing_count": 0,
      "missing_rate": 0.0
    },
    "duplicate_rows": 0,
    "duplicate_rate": 0.0
  },
  "statistics": {
    "total_outliers": 6,
    "outlier_rate": 0.06,
    "columns_with_outliers": 5,
    "mean_absolute_skewness": 0.7362,
    "max_absolute_skewness": 1.0806,
    "high_correlation_pairs": 0,
    "max_absolute_correlation": 0.5276
  },
  "target": {
    "name": "target",
    "task_type": "classification",
    "class_count": 2,
    "imbalance_ratio": 1.0,
    "minority_percentage": 50.0,
    "class_entropy": 1.0,
    "regression_stats": null
  },
  "complexity_score": 2.17,
  "complexity_detail": {
    "label": "Low",
    "normalized_components": {
      "missingness": 0.0,
      "outliers": 0.4,
      "skewness": 0.2454,
      "imbalance": 0.0,
      "dimensionality": 0.5
    },
    "weights": {
      "missingness": 0.20,
      "outliers": 0.20,
      "skewness": 0.15,
      "imbalance": 0.25,
      "dimensionality": 0.20
    }
  },
  "profiling_time_ms": 2.5
}
```

---

## 3. Metrics Specification

### A. Dataset Metadata & Canonical Hash
- `rows` ($N$): Total number of instances.
- `columns` ($K$): Total number of columns (features + target).
- `feature_count` ($N_{\text{feat}}$): Total feature columns ($K - 1$).
- `memory_bytes`: In-memory DataFrame size in bytes (`df.memory_usage(deep=True).sum()`).
- `dataset_hash`: Deterministic, OS-independent canonical dataset-content SHA-256 hash calculated over column-sorted DataFrame bytes with explicit `\n` line endings.

### B. Feature Types & Binary Recognition
- `numeric_features`: Count of continuous numerical features ($> 2$ unique values).
- `categorical_features`: Count of string/object features ($> 2$ unique values).
- `binary_features`: Count of any feature (numeric 0/1, string, or boolean) with **exactly 2 unique non-null values**.
- `boolean_features`: Count of native `bool` dtype features.
- `datetime_features`: Count of datetime features.

### C. Data Quality Metrics
- **Feature Missingness**: Computed exclusively across feature cells ($N_{\text{rows}} \times N_{\text{features}}$).
  $$\text{feature\_missing\_rate} = \frac{\text{total\_feature\_missing}}{N_{\text{rows}} \times N_{\text{features}}}$$
- **Target Missingness**: Computed on target column ($N_{\text{target\_missing}} / N_{\text{rows}}$).
- **Duplicates**: Count and rate of identical duplicate rows.

### D. Statistical Profile & Computational Complexity
- **IQR Outlier Detection**:
  $$IQR = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times IQR, \quad \text{Upper Bound} = Q_3 + 1.5 \times IQR$$
  Outlier counts are computed across numerical feature columns excluding boolean dtypes.
- **Skewness**:
  Calculated per numerical feature column using sample skewness ($g_1$).
- **Pearson Correlation**:
  Computed across numerical features using **pairwise-complete correlation** (missing values dropped per feature pair). Counts upper-triangular pairs $(i < j)$ with $|r_{ij}| \ge 0.90$.
- **Computational Complexity Considerations**:
  - Per-feature profiling (missingness, type detection, IQR outliers, skewness) runs in approximately $O(N \cdot K)$ time.
  - Pairwise Pearson correlation matrix calculation over $K$ numerical features runs in approximately $O(N \cdot K^2)$ time.

### E. Target & Task Analysis
- **Task Type Heuristic**:
  - Non-numeric dtypes (object/string/category/bool) $\implies$ `"classification"`.
  - Continuous floats with non-zero fractional parts $\implies$ `"regression"`.
  - Numeric dtypes with unique values $> 20$ OR unique-to-sample ratio $> 0.5$ (when $N \ge 10$) $\implies$ `"regression"`.
  - Otherwise $\implies$ `"classification"`.
- **Classification Metrics**:
  - `class_count`, `majority_class`, `minority_class`.
  - `imbalance_ratio`: $N_{\text{majority}} / N_{\text{minority}}$.
  - `minority_percentage`: $(N_{\text{minority}} / N_{\text{valid}}) \times 100$.
  - `class_entropy`: $H(Y) = -\sum p_i \log_2(p_i)$.

---

## 4. GENESIS DIP Complexity Score

The complexity score is an **initial engineering heuristic** (scaled from $0.0$ to $10.0$) designed to quantify overall dataset difficulty for search-space optimization.

### Formula
$$C = 10 \times \left( w_M \cdot M + w_O \cdot O + w_S \cdot S + w_I \cdot I + w_D \cdot D \right)$$

Where normalized components in $[0.0, 1.0]$ are defined as:
1. $M$ (Missingness): $\min(\text{feature\_missing\_rate} / 0.30, 1.0)$
2. $O$ (Outliers): $\min(\text{outlier\_rate} / 0.15, 1.0)$
3. $S$ (Skewness): $\min(\text{mean\_abs\_skewness} / 3.0, 1.0)$
4. $I$ (Imbalance): $\min(\max(\text{imbalance\_ratio} - 1.0, 0.0) / 19.0, 1.0)$ for classification ($0.0$ for regression)
5. $D$ (Dimensionality): $\min(\text{feature\_to\_sample\_ratio} / 0.50, 1.0)$

### Component Weights
- Missingness ($w_M$): **20%**
- Outliers ($w_O$): **20%**
- Skewness ($w_S$): **15%**
- Imbalance ($w_I$): **25%**
- Dimensionality ($w_D$): **20%**

### Categorical Difficulty Tiers
- **0.0 – 3.0**: `Low`
- **3.0 – 6.0**: `Medium`
- **6.0 – 8.0**: `High`
- **8.0 – 10.0**: `Very High`

---

## 5. Software Validation Datasets & Measured Outputs

The DIP engine was validated on 5 controlled software-validation datasets in `data/test_datasets/`:

| Dataset Filename | Task Type | Rows | Features | Feature Missing Rate | Outlier Rate | Imbalance Ratio | Complexity Score | Label | Canonical Dataset SHA-256 Hash |
|---|---|---|---|---|---|---|---|---|---|
| `01_numerical_classification.csv` | classification | 20 | 5 | 0.0000 | 0.0600 | 1.00 | **2.17** | **Low** | `cf953949b9e1838795bb00a1f9f634990f97e9a18f4d94c98ee434d1f31f01fe` |
| `02_categorical_heavy.csv` | classification | 12 | 5 | 0.0000 | 0.0000 | 1.00 | **1.67** | **Low** | `c87376d30a74467053c00316619d7fcfeb4e4b73764ebf17b627f79a8192d605` |
| `03_missing_values.csv` | classification | 10 | 4 | 0.2500 | 0.0000 | 1.00 | **3.44** | **Medium** | `13fb4dfed49021342270c2ee30fb2209660ebc5a030cbf44bd76cc58f21c69ea` |
| `04_imbalanced_classification.csv` | classification | 20 | 3 | 0.0000 | 0.1000 | 9.00 | **4.40** | **Medium** | `2567b9359765690a23ccdb83449a2cf0bdf3eeda263ef795754abc61fe70b62c` |
| `05_regression.csv` | regression | 15 | 4 | 0.0000 | 0.0000 | N/A | **1.24** | **Low** | `1ff50a62b3ac009c2bef01a30d422668d612286c56d2425278e35cf0313b87f1` |
