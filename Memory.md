# GENESIS-AI Project Memory Log

## Current Status
- **Week 2 (Baseline)**: Completed & Frozen (Commit `4068902`, DIP v1.1)
- **Week 3 (Baseline)**: Completed & Frozen (Commit `cc8cc8c`, Recommendation Engine v1.1)
- **Week 4.1 (Hardening & Research Correction Pass)**: Completed & Verified
- **Week 4.2 (Controlled Experimental Validation & Diagnosis)**: Completed & Verified
- **Week 4.2.1 (Research Analysis Correction & Reporting Hardening)**: Completed & Verified
- **Week 4.2.2 (Final Research Cleanup & Wording Hardening)**: Completed, Verified, & Documented

---

## Week 4.2.2 — Final Research Cleanup & Wording Hardening

### Implementation Status
Completed and fully verified with automated test suite (**99/99 passed**), 115 multi-seed experiment runs, synchronized research analysis, and cautious research interpretations.

### Key Research Wording & Analysis Cleanup
1. **Cautious Runtime Claims**:
   - Replaced broad claims (*"over 90% runtime savings"*) with exact aggregate wording:
     > *"GENESIS Top-K=2 reduced mean runtime by approximately 29.3% in the current development experiment (7.28s to 5.15s overall), with substantially larger reductions (over 90%) observed on specific individual datasets (such as 01, 04, and 05)."*
2. **Cautious Predictive Performance Claims**:
   - Replaced global claims (*"zero loss in predictive accuracy"*) with accurate trade-off wording:
     > *"GENESIS can substantially reduce the candidate space and, on several development datasets, maintain comparable predictive performance. However, aggressive pruning can reduce predictive performance when important candidate pipelines are excluded or when evolutionary optimization selects different configurations."*
3. **Refined Baseline Winner Retention Interpretation**:
   - Clarified 2 distinct mechanisms: (1) *Recommendation Exclusion* (model family pruned from Top-K), and (2) *Optimization Divergence* (model family included, but GA explores a different hyperparameter configuration).
4. **Historical Baseline Files Verified**:
   - `experiments/week2_dip_v1_1_results.json` (Commit `4068902`) and `experiments/week3_recommendation_results.json` (Commit `cc8cc8c`) were verified as 100% unchanged.

### Cautious Research Conclusion
> *"GENESIS demonstrates controllable candidate-space reduction. The current development experiments indicate a trade-off between search-space reduction, computational efficiency, and predictive performance. Smaller Top-K values provide greater reduction but increase the risk of excluding strong candidate pipelines. Retaining a baseline-winning pipeline does not guarantee identical performance because evolutionary optimization may select different hyperparameters. Broader benchmark evaluation (Phase 8) is required before making general claims about GENESIS-AI."*

### Pytest Execution Status
- Total tests: 99 (38 Week 2 + 31 Week 3 + 30 Week 4)
- Passed: 99
- Failed: 0
- Execution time: ~17.0s

### Summary Table — Search Cost & Candidate Reduction (115 Runs)

| Mode / Config | Total Runs | Mean Candidate Reduction | Mean Runtime | Paired Runtime Diff vs Baseline | Mean Unique Evals | Paired Unique Evals Diff vs Baseline | Mean Cache Hit Rate | Winner Retention Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BASELINE (Full Pool)** | 25 | **0.0%** | **7.28s** | **+0.00s** | **53.44** | **+0.00** | **73.28%** | **100.0%** |
| **GENESIS (Top-K=2)** | 25 | **56.0%** | **5.15s** | **-2.13s** | **45.12** | **-8.32** | **77.44%** | **52.0%** |
| **GENESIS (Top-K=3)** | 25 | **34.0%** | **5.44s** | **-1.84s** | **51.60** | **-1.84** | **74.20%** | **76.0%** |
| **GENESIS (Top-K=4)** | 25 | **12.0%** | **8.77s** | **+1.49s** | **55.48** | **+2.04** | **72.26%** | **88.0%** |
| **GENESIS (Top-K=5)** | 15 | **0.0%** | **5.78s** | **-0.26s** | **50.67** | **+0.00** | **74.67%** | **100.0%** |

### Week 5 Status
- **Week 5 has NOT been implemented.**
