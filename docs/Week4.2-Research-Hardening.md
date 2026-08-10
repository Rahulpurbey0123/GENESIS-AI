# GENESIS-AI Week 4.2.2 — Controlled Experimental Validation & Algorithmic Diagnosis (Final Research Cleanup)

## Executive Overview

Week 4.2.2 performs a final research cleanup pass over the **controlled, multi-seed (5 seeds), multi-Top-K (2, 3, 4, 5) empirical validation** of the **GENESIS-AI Evolutionary Pipeline Optimization Engine** across 5 software validation datasets.

The investigation evaluates the core research question:
> *"Can dataset-guided candidate-space reduction reduce AutoML search cost while maintaining comparable predictive performance?"*

This phase enforces strict experimental controls, corrected cache-hit-rate calculations, task-specific metric separation (classification Macro-F1 vs regression RMSE), candidate inclusion diagnostics (`baseline_best_in_genesis_top_k`), and unbiased statistical reporting without manipulating algorithms, metrics, seeds, or evaluation results.

---

## 1. Experimental Design & 115-Run Breakdown

To ensure rigorous paired comparison between `GENESIS` (experimental group) and `BASELINE` (control group), every matched run pair enforces:

- **Same Dataset & Target Column**: Identical raw dataset input and target column choice.
- **Same Train / Validation / Test Data Split**: Identical train ($60\%$), validation ($20\%$), and isolated test ($20\%$) partitions.
- **Same Random Seed**: Matched seed array `[42, 123, 456, 789, 2024]` propagated across `train_test_split`, initial population generation, selection, crossover, and mutation.
- **Same Task-Specific Metrics**: Macro-F1 for classification; $-RMSE$ for regression.
- **Same GA Configuration**:
  - `population_size`: 20
  - `generations`: 10
  - `max_evaluations`: 200
  - `crossover_rate`: 0.80
  - `mutation_rate`: 0.10
  - `pipeline_mutation_rate`: 0.10
  - `elite_size`: 2
  - `tournament_size`: 3
  - `fitness_cache`: enabled
- **Only Variable**: Initial candidate space restriction (`GENESIS` Top-K pool vs `BASELINE` full compatible candidate pool).

### 115 Total Runs Breakdown
- **BASELINE**: 5 datasets $\times$ 5 seeds = **25 runs**
- **GENESIS Top-K=2**: 5 datasets $\times$ 5 seeds = **25 runs**
- **GENESIS Top-K=3**: 5 datasets $\times$ 5 seeds = **25 runs**
- **GENESIS Top-K=4**: 5 datasets $\times$ 5 seeds = **25 runs**
- **GENESIS Top-K=5**: 3 datasets (`01`, `04`, `05` with 5 compatible candidates) $\times$ 5 seeds = **15 runs**
  - *(Note: Datasets `02_categorical_heavy` and `03_missing_values` have 4 compatible pipelines post-filtering, so `top_k=5` caps at 4 and is omitted to prevent duplicate runs).*
- **Total**: $25 + 25 + 25 + 25 + 15 = 115$ total experiment runs.

---

## 2. Corrected Cache Hit Rate Formula & Metrics Separation

### Corrected Cache Hit Rate Formula
The cache hit rate measures the proportion of evaluation requests served by the lookup cache:

$$\text{total\_evaluation\_requests} = \text{unique\_evaluations} + \text{cache\_hits}$$

$$\text{cache\_hit\_rate} = \frac{\text{cache\_hits}}{\text{total\_evaluation\_requests}} = \frac{\text{cache\_hits}}{\text{unique\_evaluations} + \text{cache\_hits}}$$

This formula guarantees that $0.0 \le \text{cache\_hit\_rate} \le 1.0$ (e.g. $175 / 200 = 0.875 = 87.5\%$).

### Task-Specific Predictive Metric Separation
Classification metrics (Macro-F1, higher is better) and Regression metrics (RMSE, lower is better) are analyzed and reported separately. They are NEVER combined into a single predictive performance average or difference.

---

## 3. Scientific Baseline Winner Retention Interpretation

The research diagnostic flag `baseline_best_in_genesis_top_k` records whether `BASELINE`'s winning model family was included in `GENESIS`'s Top-K candidate pool.

> [!IMPORTANT]
> **Correct Scientific Interpretation**: Retaining the BASELINE-winning pipeline within Top-K indicates that the recommendation stage did not exclude the BASELINE winner. However, this does **NOT** guarantee identical predictive performance, because evolutionary optimization may select a different hyperparameter configuration.

Two distinct mechanisms affect performance:
1. **Recommendation Exclusion**: The BASELINE-winning model family is not in Top-K; `GENESIS` cannot explore it.
2. **Optimization Divergence**: The BASELINE-winning model family is in Top-K, but GA evolutionary search explores a different hyperparameter configuration.

---

## 4. Overall Search Cost & Candidate Reduction Results

Aggregated search cost metrics across all 115 runs (`experiments/week4_analysis_summary.json`):

| Mode / Config | Total Runs | Mean Candidate Reduction | Mean Runtime | Paired Runtime Diff vs Baseline | Mean Unique Evals | Paired Unique Evals Diff vs Baseline | Mean Cache Hit Rate | Winner Retention Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BASELINE (Full Pool)** | 25 | **0.0%** | **7.28s** | **+0.00s** | **53.44** | **+0.00** | **73.28%** | **100.0%** |
| **GENESIS (Top-K=2)** | 25 | **56.0%** | **5.15s** | **-2.13s** | **45.12** | **-8.32** | **77.44%** | **52.0%** |
| **GENESIS (Top-K=3)** | 25 | **34.0%** | **5.44s** | **-1.84s** | **51.60** | **-1.84** | **74.20%** | **76.0%** |
| **GENESIS (Top-K=4)** | 25 | **12.0%** | **8.77s** | **+1.49s** | **55.48** | **+2.04** | **72.26%** | **88.0%** |
| **GENESIS (Top-K=5)** | 15 | **0.0%** | **5.78s** | **-0.26s** | **50.67** | **+0.00** | **74.67%** | **100.0%** |

---

## 5. Task-Specific Predictive Performance Results

### Classification Predictive Performance (Macro-F1: Higher is Better)

Across 4 classification datasets $\times$ 5 seeds = 20 runs per mode (10 for Top-K=5):

| Mode / Config | Classification Runs | Mean F1 | Std F1 | Median F1 | Mean F1 Diff vs Baseline | Median F1 Diff vs Baseline |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BASELINE** | 20 | **0.6000** | **0.3670** | **0.6667** | **+0.0000** | **0.0000** |
| **GENESIS (Top-K=2)** | 20 | **0.5192** | **0.3395** | **0.3667** | **-0.0808** | **0.0000** |
| **GENESIS (Top-K=3)** | 20 | **0.5725** | **0.3530** | **0.5000** | **-0.0275** | **0.0000** |
| **GENESIS (Top-K=4)** | 20 | **0.6517** | **0.3277** | **0.7000** | **+0.0517** | **0.0000** |
| **GENESIS (Top-K=5)** | 10 | **0.7000** | **0.3642** | **0.8666** | **+0.0000** | **0.0000** |

### Regression Predictive Performance (RMSE: Lower is Better)

Across 1 regression dataset $\times$ 5 seeds = 5 runs per mode:

| Mode / Config | Regression Runs | Mean RMSE | Std RMSE | Median RMSE | Mean RMSE Diff vs Baseline | Relative RMSE Diff % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BASELINE** | 5 | **8247.34** | **970.43** | **7886.09** | **+0.00** | **+0.00%** |
| **GENESIS (Top-K=2)** | 5 | **8247.34** | **970.43** | **7886.09** | **+0.00** | **+0.00%** |
| **GENESIS (Top-K=3)** | 5 | **8247.34** | **970.43** | **7886.09** | **+0.00** | **+0.00%** |
| **GENESIS (Top-K=4)** | 5 | **8247.34** | **970.43** | **7886.09** | **+0.00** | **+0.00%** |
| **GENESIS (Top-K=5)** | 5 | **8247.34** | **970.43** | **7886.09** | **+0.00** | **+0.00%** |

*(Note: For RMSE, negative difference means GENESIS improved/lower RMSE; positive difference means GENESIS performed worse/higher RMSE).*

---

## 6. Per-Dataset Experimental Summary

- **`01_numerical_classification.csv`**: `GENESIS Top-K=2` reduced mean runtime from **7.95s $\rightarrow$ 0.35s** (a **95.6% runtime reduction**) and unique evaluations from **56.0 $\rightarrow$ 27.4**. Test F1 increased from 0.4000 to 0.5800 on Top-K=2.
- **`02_categorical_heavy.csv`**: `GENESIS Top-K=2` pruned `SVC`. Under seed 2024, `SVC` was BASELINE's winner; excluding it reduced Top-K=2 test F1 score. Expanding `Top-K` to `3` included `SVC`, raising winner retention to **80.0%** and test F1 score to **0.7167**.
- **`03_missing_values.csv`**: `GENESIS Top-K=2` reduced mean runtime from **12.32s $\rightarrow$ 3.72s** (**69.8% runtime reduction**) with higher mean test F1 (0.2666 vs 0.2000).
- **`04_imbalanced_classification.csv`**: `GENESIS Top-K=2` reduced mean runtime from **6.76s $\rightarrow$ 0.41s** (**93.9% runtime reduction**) with 100% test F1 retention (1.0000).
- **`05_regression.csv`**: `GENESIS Top-K=2` reduced mean runtime from **3.43s $\rightarrow$ 0.24s** (**93.0% runtime reduction**) and unique evaluations from **37.2 $\rightarrow$ 25.4** with 100% test RMSE retention (8247.34, $R^2 = 0.995$).

---

## 7. Cautious Evidence-Based Research Conclusions

GENESIS demonstrates controllable candidate-space reduction. The current development experiments indicate a trade-off between search-space reduction, computational efficiency, and predictive performance:

1. **Runtime Efficiency**: GENESIS Top-K=2 reduced mean runtime by approximately 29.3% in the current aggregate development experiment (7.28s to 5.15s overall), with substantially larger reductions (over 90%) observed on specific individual datasets (such as `01`, `04`, and `05`).
2. **Predictive Performance Trade-Off**: GENESIS can substantially reduce the candidate space and, on several development datasets, maintain comparable predictive performance. However, aggressive pruning can reduce predictive performance when important candidate pipelines are excluded or when evolutionary optimization selects different configurations.
3. **Top-K Selection Risk**: Smaller Top-K values (e.g. `top_k=2`) provide greater candidate reduction ($56.0\%$) and faster runtime, but increase the risk of excluding strong candidate pipelines. Retaining a baseline-winning pipeline does not guarantee identical performance because evolutionary optimization may select different hyperparameters. Broader benchmark evaluation (Phase 8) is required before making general claims about GENESIS-AI.

---

## 8. Output Artifacts

- [`experiments/analyze_week4_results.py`](file:///d:/GENESIS-AI/experiments/analyze_week4_results.py): Corrected research analysis pipeline.
- [`experiments/week4_analysis_summary.json`](file:///d:/GENESIS-AI/experiments/week4_analysis_summary.json): Synchronized numerical source of truth.
- [`experiments/week4_analysis_summary.csv`](file:///d:/GENESIS-AI/experiments/week4_analysis_summary.csv): Tabular CSV summary.
- [`tests/test_optimization_research_validation.py`](file:///d:/GENESIS-AI/tests/test_optimization_research_validation.py): Research unit tests (99/99 passing).
