# GENESIS-AI Week 7 — Research Evaluation & Scientific Validation Report (v1.4 Final)

## 1. Objective
The objective of Week 7 is to experimentally evaluate whether the GENESIS-AI framework—combining Dataset Intelligence Profiling (DIP v1.1), Intelligent Recommendation, and Evolutionary Optimization—achieves competitive predictive performance and improved candidate search efficiency compared against established baseline search and ablation methods.

The central research question guiding Week 7 is:
> *"Does dataset-aware search-space reduction through DIP, combined with recommendation and evolutionary optimization, find competitive ML pipelines efficiently and reproducibly?"*

---

## 2. Research Questions
- **RQ1**: Does the full GENESIS-AI pipeline achieve competitive predictive performance against selected baselines?
- **RQ2**: Does DIP-guided search reduce candidate evaluations while maintaining competitive performance?
- **RQ3**: Does evolutionary optimization improve solution quality compared with recommendation-only selection?
- **RQ4**: Are results stable across repeated runs?

---

## 3. Hypotheses
- **H1**: GENESIS-AI achieves competitive predictive performance.
- **H2**: DIP-guided search evaluates fewer candidate pipelines while maintaining competitive predictive performance.
- **H3**: Evolutionary optimization improves solution quality compared with recommendation-only selection.
- **H4**: GENESIS-AI produces stable results across repeated runs.

*Note: All hypotheses are treated as falsifiable scientific hypotheses and evaluated strictly against empirical evidence without post-hoc thresholds.*

---

## 4. Baselines and Ablation Matrix
The evaluation matrix consists of 5 comparison methods (A–E) designed for complete component isolation:

| Code | Method Identifier | Component Breakdown | Search Space Restriction | Optimization | DIP Generator Call | Top-K Pool Equivalence |
|:---:|---|---|---|---|:---:|:---:|
| **A** | `method_a_full_genesis` | Full GENESIS-AI | Top-K=2 Recommendations (DIP Profile) | Genetic Algorithm | **YES** | Reference |
| **B** | `method_b_without_dip` | Without DIP | Top-K=2 Recommendations (Neutral Profile) | Genetic Algorithm | **NO** | Same $K=2$ |
| **C** | `method_c_without_recommendation` | Without Recommendation | All Compatible Pipelines (No Pruning) | Genetic Algorithm | **YES** | All Compatible |
| **D** | `method_d_recommendation_only` | Recommendation Only | Top-K=2 Recommended Default Pipelines | Default Hyperparameters (No GA) | **YES** | Same $K=2$ |
| **E** | `method_e_unguided_baseline` | Unguided Baseline | All Task Pipelines & Grids | Random Search (Equal Budget = 200) | **NO** | Full Grid |

### Component Ablation Isolation Proofs
- **DIP Ablation (A vs B)**: Method B executes through the **exact same `RecommendationEngine` pathway** (`RecommendationEngine.recommend_from_dip`) as Method A, but passes a synthetic **neutral DIP profile** (`create_neutral_dip`). Method B contains ZERO dataset intelligence profiling, ZERO usage of dataset-specific feature signals, and ZERO calls to `generate_dip()`.
- **GA Evolutionary Optimization Ablation (A vs D)**: Method D queries `RecommendationEngine` with the **exact same DIP profile and candidate pool size** (`top_k = self.config.top_k`, $K=2$) as Method A. Method D evaluates the default hyperparameter configurations across the Top-K pool on Validation splits, selecting the best default model without GA search. Recommendation candidate pool and DIP profile are held strictly constant between A and D.

---

## 5. Dataset Selection
Experiments were evaluated across the 5 standard development datasets:

1. `01_numerical_classification.csv`: Clean numerical binary classification task (`target`).
2. `02_categorical_heavy.csv`: Categorical-heavy binary classification task (`subscribed`).
3. `03_missing_values.csv`: Missing values classification task (`label`).
4. `04_imbalanced_classification.csv`: Class-imbalanced classification task (`is_fraud`).
5. `05_regression.csv`: Continuous target house price regression task (`price`).

---

## 6. Primary Evaluation Metrics & Directionality
- **Classification Tasks**: Primary Metric is **Macro F1 Score** (`f1`) and Accuracy (`accuracy`). Optimization Direction: **HIGHER IS BETTER**.
- **Regression Tasks**: Primary Metric is **Root Mean Squared Error** (`rmse`), Mean Absolute Error (`mae`), and $R^2$. Optimization Direction: **LOWER IS BETTER** for RMSE/MAE, higher for $R^2$.
- **Search Efficiency Metrics**: Candidate pipelines evaluated (`evaluations_used`), search space reduction ratio (`candidate_space_reduction`), and total runtime in seconds (`runtime_seconds`).

*Note: Classification metrics (Macro F1) and Regression metrics (RMSE) are validated through a dedicated metric validation layer (`validate_comparable_observations`) and kept strictly separate in all statistical analyses, p-value calculations, and summary reports. No single combined predictive score or mixed-metric p-value is generated.*

---

## 7. Experimental Protocol & Decision Rules
- **Data Partitions**: Strict seed-governed Train (60%), Validation (20%), and isolated Test (20%) splits across all methods per seed.
- **Fixed Random Seeds**: 5 fixed random seeds (`[42, 123, 456, 789, 2024]`).
- **Comparable Budget**: Maximum evaluation budget of 200 candidate evaluations per search run for methods A, B, C, and E.
- **Isolated Test Evaluation**: Final best pipeline per run is evaluated strictly **once** post-search on the held-out test split.
- **DIP Independence Proof**: Method B and Method E have ZERO calls to `generate_dip()` or DIP modules.

### Hypothesis Decision Principles (Task-Separated & Direction-Aware)
- **H1 Decision Rule**: Primary comparison evaluates Method A (Full GENESIS-AI) vs Method E (Unguided Baseline) separately per task metric (Classification Macro F1: higher is better; Regression RMSE: lower is better). Status is `SUPPORTED` if Method A achieves statistically significant performance improvement over Method E ($p < 0.05$) without statistically significant degradation on either task. Status is `NOT SUPPORTED` if Method A is statistically significantly degraded ($p < 0.05$) compared to Method E. Status is `INCONCLUSIVE` if performance differences are statistically non-significant ($p \ge 0.05$) or yield mixed findings across dataset types. Dataset-level comparisons and matched counts are reported descriptively, but do not independently determine H1 support.
- **H2 Dual-Condition Decision Rule**:
  - *Condition 1 (Efficiency)*: Candidate evaluations of Method A are significantly lower than Method B ($E_A < E_B$ with paired t-test $p < 0.05$).
  - *Condition 2 (Task-Separated Performance Maintenance)*: Classification Macro F1 ($S_A \ge S_B$ or no statistically significant F1 degradation $p \ge 0.05$) AND Regression RMSE ($RMSE_A \le RMSE_B$ or no statistically significant RMSE increase $p \ge 0.05$).
  - *Decision Priority*: Status is `NOT SUPPORTED` if predictive performance is significantly degraded ($p < 0.05$) on either task type regardless of search efficiency. Status is `SUPPORTED` if BOTH efficiency reduction and performance maintenance pass. Status is `INCONCLUSIVE` if candidate evaluation reduction or predictive performance differences are statistically non-significant ($p \ge 0.05$).
- **H3 Decision Rule**: Method A must achieve statistically significant performance improvement over Method D evaluated separately on Classification Macro F1 ($p < 0.05$, higher F1) or Regression RMSE ($p < 0.05$, lower RMSE) without statistically significant degradation on either task. Status is `SUPPORTED` if significant improvement exists and zero degradation exists. Status is `NOT SUPPORTED` if statistically significant degradation exists on either task ($p < 0.05$). Status is `INCONCLUSIVE` if score differences are small or statistically non-significant ($p \ge 0.05$).
- **H4 Decision Rule**: Evaluated separately for Classification Stability ($CV$ of F1 across classification datasets) and Regression Stability ($CV$ of RMSE across regression datasets). Status is reported descriptively without post-hoc thresholds.
- **Statistical Fallback Rule**: When a statistical test cannot be validly performed (e.g., zero variance across paired runs or single observation splits), the evaluation reports the observed mean difference descriptively but does not interpret raw mean direction as statistically significant improvement or degradation. Unsupported statistical inference is strictly prevented and hypothesis status remains `INCONCLUSIVE`.

---

## 8. Empirical Results Summary

### Aggregated Performance across All 125 Benchmark Runs

| Dataset Filename | Task Type | Method Code | Method Name | Primary Metric | Direction | Mean Score | Std Score | Best Score | Worst Score | Mean Evals | Mean Time (s) |
|---|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `01_numerical_classification.csv` | Classification | **A** | Full GENESIS-AI | F1 | Higher | **0.5800** | 0.2352 | 0.7333 | 0.2000 | **27.4** | **0.26s** |
| `01_numerical_classification.csv` | Classification | **B** | Without DIP | F1 | Higher | 0.4200 | 0.2049 | 0.7333 | 0.2000 | 52.8 | 2.99s |
| `01_numerical_classification.csv` | Classification | **C** | Without Recommendation | F1 | Higher | 0.4000 | 0.3266 | 0.7333 | 0.0000 | 56.0 | 7.79s |
| `01_numerical_classification.csv` | Classification | **D** | Recommendation Only | F1 | Higher | 0.5333 | 0.2198 | 0.7333 | 0.2000 | 2.0 | 0.06s |
| `01_numerical_classification.csv` | Classification | **E** | Unguided Baseline | F1 | Higher | 0.4933 | 0.2994 | 0.7333 | 0.0000 | 200.0 | 30.54s |
| `02_categorical_heavy.csv` | Classification | **A** | Full GENESIS-AI | F1 | Higher | 0.2300 | 0.1440 | 0.4000 | 0.0000 | 88.6 | 30.77s |
| `02_categorical_heavy.csv` | Classification | **B** | Without DIP | F1 | Higher | 0.6333 | 0.3755 | 1.0000 | 0.2500 | **39.6** | **3.27s** |
| `02_categorical_heavy.csv` | Classification | **C** | Without Recommendation | F1 | Higher | **0.8000** | 0.1826 | 1.0000 | 0.6667 | 46.4 | 7.06s |
| `02_categorical_heavy.csv` | Classification | **D** | Recommendation Only | F1 | Higher | 0.4000 | 0.3354 | 1.0000 | 0.2500 | 2.0 | 0.57s |
| `02_categorical_heavy.csv` | Classification | **E** | Unguided Baseline | F1 | Higher | **0.8667** | 0.1826 | 1.0000 | 0.6667 | 200.0 | 15.41s |
| `03_missing_values.csv` | Classification | **A** | Full GENESIS-AI | F1 | Higher | **0.2666** | 0.1491 | 0.3333 | 0.0000 | **56.2** | 2.95s |
| `03_missing_values.csv` | Classification | **B** | Without DIP | F1 | Higher | 0.2000 | 0.1826 | 0.3333 | 0.0000 | **56.2** | **2.82s** |
| `03_missing_values.csv` | Classification | **C** | Without Recommendation | F1 | Higher | 0.2000 | 0.1826 | 0.3333 | 0.0000 | 68.8 | 9.07s |
| `03_missing_values.csv` | Classification | **D** | Recommendation Only | F1 | Higher | 0.3333 | 0.0000 | 0.3333 | 0.3333 | 2.0 | 0.11s |
| `03_missing_values.csv` | Classification | **E** | Unguided Baseline | F1 | Higher | 0.2000 | 0.1826 | 0.3333 | 0.0000 | 200.0 | 13.52s |
| `04_imbalanced_classification.csv` | Classification | **A** | Full GENESIS-AI | F1 | Higher | **1.0000** | 0.0000 | 1.0000 | 1.0000 | **28.0** | **0.24s** |
| `04_imbalanced_classification.csv` | Classification | **B** | Without DIP | F1 | Higher | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 57.6 | 2.64s |
| `04_imbalanced_classification.csv` | Classification | **C** | Without Recommendation | F1 | Higher | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 58.8 | 4.47s |
| `04_imbalanced_classification.csv` | Classification | **D** | Recommendation Only | F1 | Higher | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 2.0 | 0.04s |
| `04_imbalanced_classification.csv` | Classification | **E** | Unguided Baseline | F1 | Higher | **1.0000** | 0.0000 | 1.0000 | 1.0000 | 200.0 | 12.97s |
| `05_regression.csv` | Regression | **A** | Full GENESIS-AI | RMSE | Lower | **8247.34** | 1084.97 | 7356.69 | 10131.51 | **25.4** | **0.18s** |
| `05_regression.csv` | Regression | **B** | Without DIP | RMSE | Lower | **8247.34** | 1084.97 | 7356.69 | 10131.51 | 31.2 | 1.69s |
| `05_regression.csv` | Regression | **C** | Without Recommendation | RMSE | Lower | **8247.34** | 1084.97 | 7356.69 | 10131.51 | 37.2 | 2.63s |
| `05_regression.csv` | Regression | **D** | Recommendation Only | RMSE | Lower | **8247.34** | 1084.97 | 7356.69 | 10131.51 | 2.0 | 0.03s |
| `05_regression.csv` | Regression | **E** | Unguided Baseline | RMSE | Lower | **8247.34** | 1084.97 | 7356.69 | 10131.51 | 200.0 | 10.75s |

---

## 9. Ablation Study Breakdown (Task Metrics Separated)

| Method Code | Method Name | Has DIP | Has Rec | Has Opt | Mean Classification F1 | Mean Regression RMSE | Mean Candidate Evals | Mean Time (s) | Relative Search Efficiency Gain vs Baseline |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Full GENESIS-AI | Yes | Yes | Yes | **0.5192** | **8247.34** | **45.1** | **6.88s** | **+77.4%** |
| **B** | Without DIP | No | Yes | Yes | 0.5633 | 8247.34 | 47.5 | 2.68s | +76.3% |
| **C** | Without Recommendation | Yes | No | Yes | 0.6000 | 8247.34 | 53.4 | 6.20s | +73.3% |
| **D** | Recommendation Only | Yes | Yes | No | 0.5667 | 8247.34 | **2.0** | **0.16s** | **+99.0%** |
| **E** | Unguided Baseline | No | No | No | 0.6400 | 8247.34 | 200.0 | 16.64s | 0.0% (Control) |

---

## 10. Statistical Hypothesis Evaluation

### H1: GENESIS-AI achieves competitive predictive performance.
- **Status**: **INCONCLUSIVE**
- **Rationale**: Method A matched or outperformed Method E on 4 out of 5 datasets (`01_numerical_classification.csv`: F1 0.5800 vs 0.4933; `03_missing_values.csv`: F1 0.2666 vs 0.2000; `04_imbalanced_classification.csv`: F1 1.0000 vs 1.0000; `05_regression.csv`: RMSE 8247.34 vs 8247.34), yielding mixed performance findings across dataset types. Score differences between Method A and baseline Method E across task metrics were statistically non-significant ($p \ge 0.05$) [Classification Macro F1: A=0.5192 vs E=0.6400 (diff=-0.1208, $t=-1.548$, $p=0.1382$); Regression RMSE: A=8247.34 vs E=8247.34 (diff=+0.0000)]. Evaluated strictly using task-separated statistical paired tests without post-hoc thresholds.

### H2: DIP-guided search evaluates fewer candidate pipelines while maintaining competitive predictive performance.
- **Status**: **INCONCLUSIVE**
- **Rationale**: Method A evaluated an average of **45.1** candidate pipelines per run compared to **47.5** for Method B (Without DIP), **53.4** for Method C (Without Recommendation), and **200.0** for Method E (Unguided Baseline). Method A evaluated fewer candidates on average than Method B, but the candidate evaluation reduction was not statistically significant ($p = 0.7078 \ge 0.05$) across the 5 development datasets. Classification Macro F1 (A=0.5192 vs B=0.5633, diff=-0.0442, $t=-0.691$, $p=0.4977$) and Regression RMSE (A=8247.34 vs B=8247.34, diff=+0.0000) performance were both non-significantly different.

### H3: Evolutionary optimization improves solution quality compared with recommendation-only selection.
- **Status**: **INCONCLUSIVE**
- **Rationale**: Evaluated strictly separately per task metric. Method A (DIP + Recommendation + Top-K=2 candidate pool + Evolutionary Optimization GA) vs Method D (DIP + Recommendation + Top-K=2 candidate pool + Default Hyperparameters / NO GA). Recommendation candidate pool and DIP profile are held strictly constant between A and D so the isolated variable is GA optimization. Evaluated separately in Python implementation: For Classification F1, Method A (F1 0.5192) vs Method D (F1 0.5667) had diff=-0.0475, $t=-1.257$, $p = 0.2241 \ge 0.05$. For Regression RMSE, Method A (RMSE 8247.34) vs Method D (RMSE 8247.34) had identical scores ($p=1.0$). Score differences between Method A (GA) and Method D (Recommendation Only) across datasets were small or statistically non-significant ($p \ge 0.05$).

### H4: GENESIS-AI produces stable results across repeated runs.
- **Status**: **INCONCLUSIVE**
- **Rationale**: Evaluated separately for Classification and Regression. Classification Stability: Mean F1 $CV = 0.3977$ (`01`: 0.4056; `02`: 0.6263; `03`: 0.5590; `04`: 0.0000). Regression Stability: Mean RMSE $CV = 0.1316$ (`05`: 0.1316). Task-specific run-to-run variability is primarily driven by small development dataset evaluation splits (14–300 rows) and stochastic GA seed initialization.

---

## 11. Experimental Limitations & Regression Dataset Note
1. **Small Regression Dataset Limitation**: `05_regression.csv` contains only ~15 observations. As a consequence, the 60/20/20 train/validation/test split produces tiny evaluation partitions (9 train rows, 3 validation rows, 3 test rows). All 5 search methods evaluate small candidate configurations or converge to the exact same regression model (`LinearRegression`), yielding identical RMSE values (8247.34) across seeds. Therefore, `05_regression.csv` provides limited discrimination between search strategies.
2. **Small Dataset Sample Variance**: Small sample sizes across the 5 development datasets (14–300 rows) generate high performance variance across random seed splits.

---

## 12. Reproducibility Statement
All 125 benchmark runs were executed with fixed random seeds (`[42, 123, 456, 789, 2024]`), standardized train/validation/test dataset splits, and isolated post-search test set evaluations. Method B and Method E were verified to execute with ZERO DIP calls (`generate_dip`). Raw observations are preserved in `experiments/week7_benchmark_results.json`, `experiments/week7_benchmark_results_v1_2.json`, `experiments/week7_benchmark_results_v1_3.json`, and `experiments/week7_benchmark_results_v1_4.json`, summary metrics in `experiments/week7_summary.csv`, `experiments/week7_summary_v1_2.csv`, `experiments/week7_summary_v1_3.csv`, and `experiments/week7_summary_v1_4.csv`, and ablation breakdowns in `experiments/week7_ablation_results.json`, `experiments/week7_ablation_results_v1_2.json`, `experiments/week7_ablation_results_v1_3.json`, and `experiments/week7_ablation_results_v1_4.json`.
