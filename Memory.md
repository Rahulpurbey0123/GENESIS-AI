# GENESIS-AI Project Memory Log

## Current Status
- **Week 2 (Baseline)**: Completed & Frozen (Commit `4068902`, DIP v1.1)
- **Week 3 (Baseline)**: Completed & Frozen (Commit `cc8cc8c`, Recommendation Engine v1.1)
- **Week 4 (Baseline)**: Completed & Frozen (Commit `5e52abf`, Evolutionary Optimization v1.2)
- **Week 4.2.2 (Research Cleanup & Wording Hardening)**: Completed & Verified
- **Week 5 (Explainability & Model Insight Engine)**: Completed, Corrected, & Hardened

---

## Week 5 — Final Research Correction & Hardening Pass

### Implementation Status
Completed and fully verified with automated test suite (**132/132 passed**), 5-dataset controlled experiment runner (`experiments/generate_explainability_results.py`), structured output JSON (`experiments/week5_explainability_results.json`), and comprehensive technical documentation (`docs/Week5-Explainability.md`).

### Key Methodological Decisions & Bug Fixes
1. **Permutation Importance Metric Enforcement**:
   - Classification strictly evaluates macro F1 (`custom_f1_macro_scorer`) and reports metric `f1_macro`. Removed silent fallback to `accuracy`.
   - Regression strictly evaluates negative RMSE (`custom_neg_rmse_scorer`) and reports metric `neg_root_mean_squared_error`.
2. **Local Attribution Scientific Integrity**:
   - Eliminated artificial local contributions (`importance * feature_value`) for `permutation_importance` and `native_tree` methods.
   - `shap_tree` $\rightarrow$ global + local attributions.
   - `linear_coefficients` $\rightarrow$ global + local linear contributions ($w_i \cdot x_{trans, i}$).
   - `permutation_importance` $\rightarrow$ global importances only (`local_explanations = []` + warning).
   - `native_tree` $\rightarrow$ global importances only (`local_explanations = []` + warning).
3. **Local Row Alignment & Index Bug Fix**:
   - Corrected local prediction retrieval in `local.py`: predictions and targets use exact original row index `orig_row_idx`.
   - Refactored callback extractor signature to `contribution_extractor_fn(orig_row_idx)`.
4. **Pytest Execution Status**:
   - Total tests: **132** (38 Week 2 + 31 Week 3 + 30 Week 4 + 33 Week 5)
   - Passed: **132**
   - Failed: 0
   - Execution time: ~36.6s

---

## Week 4.2.2 — Final Research Cleanup & Wording Hardening

### Implementation Status
Completed and fully verified with automated test suite (**99/99 passed**), 115 multi-seed experiment runs, synchronized research analysis, and cautious research interpretations.

### Cautious Research Conclusion
> *"GENESIS demonstrates controllable candidate-space reduction. The current development experiments indicate a trade-off between search-space reduction, computational efficiency, and predictive performance. Smaller Top-K values provide greater reduction but increase the risk of excluding strong candidate pipelines. Retaining a baseline-winning pipeline does not guarantee identical performance because evolutionary optimization may select different hyperparameters. Broader benchmark evaluation (Phase 8) is required before making general claims about GENESIS-AI."*

---

## Next Action
Week 5 research correction is complete and fully verified. Awaiting project-owner review. Week 6 has NOT been implemented.
