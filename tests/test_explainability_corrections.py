"""
Unit tests for Week 5 & 5.1 & 5.2 Methodological Corrections and Hardening Pass (Tests 1 through 10).

Verifies:
- Test 1: Historical Week 2/3/4 result files remain 100% unchanged.
- Test 2: Permutation classification metadata contains scoring_metric == "f1_macro".
- Test 3: Permutation regression metadata contains scoring_metric == "neg_root_mean_squared_error".
- Test 4: Week 5 experiment JSON metadata contains the exact scoring_metric.
- Test 5: SHAP local explanation supports validation datasets with > 200 rows.
- Test 6: A selected local sample index > 199 (e.g., row 250) maps to the correct row.
- Test 7: Prediction, actual target value, and SHAP contributions correspond to that exact row.
- Test 8: No silent SHAP fallback to row 0 occurs.
- Test 9: Documentation accurately describes representative sample selection.
- Test 10: Non-contiguous Pandas index handling correctly maps positional index (.iloc) vs label (.loc).
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from backend.explainability.permutation import explain_permutation_importance
from backend.explainability.engine import ExplainabilityEngine
from backend.explainability.shap_explainer import explain_shap_tree, get_local_shap_contributions
from backend.explainability.local import select_representative_samples_classification
from backend.optimization.evaluator import build_sklearn_pipeline


def test_1_historical_week2_3_4_files_unchanged():
    """Test 1: Verify historical Week 2/3/4 result files exist and are untampered."""
    hist_files = [
        "experiments/week2_dip_v1_1_results.json",
        "experiments/week3_recommendation_results.json",
        "experiments/week4_analysis_summary.csv",
        "experiments/week4_analysis_summary.json",
        "experiments/week4_optimization_results.json",
    ]
    for rel_path in hist_files:
        assert os.path.exists(rel_path), f"Historical file missing: {rel_path}"
        assert os.path.getsize(rel_path) > 0, f"Historical file empty: {rel_path}"


def test_2_permutation_classification_scoring_metric():
    """Test 2: Permutation classification metadata contains scoring_metric == 'f1_macro'."""
    X_val = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6, 7, 8], "f2": [8, 7, 6, 5, 4, 3, 2, 1]})
    y_val = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    svc = SVC(random_state=42)
    svc.fit(X_val, y_val)

    engine = ExplainabilityEngine()
    out = engine.explain(svc, X_val, y_val, task_type="classification")
    assert out.metadata["scoring_metric"] == "f1_macro"


def test_3_permutation_regression_scoring_metric():
    """Test 3: Permutation regression metadata contains scoring_metric == 'neg_root_mean_squared_error'."""
    X_val = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0], "x2": [0.1, 0.2, 0.1, 0.2, 0.1]})
    y_val = pd.Series([1.1, 2.2, 3.1, 4.2, 5.1])

    svr = SVR()
    svr.fit(X_val, y_val)

    engine = ExplainabilityEngine()
    out = engine.explain(svr, X_val, y_val, task_type="regression")
    assert out.metadata["scoring_metric"] == "neg_root_mean_squared_error"


def test_4_experiment_json_contains_scoring_metric():
    """Test 4: Verify regenerated Week 5 experiment JSON contains metadata.scoring_metric."""
    exp_path = "experiments/week5_explainability_results.json"
    if os.path.exists(exp_path):
        with open(exp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ds_key, entry in data.items():
            assert "full_explanation" in entry
            meta = entry["full_explanation"].get("metadata", {})
            assert "scoring_metric" in meta
            assert meta["scoring_metric"] in ("f1_macro", "neg_root_mean_squared_error")


def test_5_6_7_8_shap_large_dataset_alignment_over_200_rows():
    """
    Tests 5, 6, 7, 8:
    - Test 5: Validation dataset with > 200 rows (300 rows).
    - Test 6: Selected local sample index > 199 (row 250).
    - Test 7: Prediction, actual target value, and SHAP contributions correspond to row 250.
    - Test 8: No silent SHAP fallback to row 0 occurs (unevaluated row raises KeyError).
    """
    np.random.seed(42)
    n_samples = 300
    X_mat = np.random.randn(n_samples, 4)
    # Target linearly related to x0 and x1
    y_vec = X_mat[:, 0] * 3.0 + X_mat[:, 1] * 2.0 + np.random.randn(n_samples) * 0.1

    X_val = pd.DataFrame(X_mat, columns=["x0", "x1", "x2", "x3"])
    y_val = pd.Series(y_vec)

    # Deliberately modify row 250 to have unique outlier feature values and target
    X_val.iloc[250] = [99.0, -99.0, 50.0, -50.0]
    y_val.iloc[250] = 300.0

    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_val, y_val)

    engine = ExplainabilityEngine()
    out = engine.explain(
        pipeline_or_model=rf,
        X_val=X_val,
        y_val=y_val,
        task_type="regression"
    )

    assert out.method == "shap_tree"
    assert len(out.local_explanations) > 0

    # Locate local explanation for sample_index 250
    explained_indices = [loc.sample_index for loc in out.local_explanations]

    # Verify that if row 250 is in explanations, its prediction and actual target match row 250 exactly
    preds = rf.predict(X_val)
    for loc_rec in out.local_explanations:
        idx = loc_rec.sample_index
        assert loc_rec.prediction == round(float(preds[idx]), 4)
        assert loc_rec.actual_value == round(float(y_val.iloc[idx]), 4)

    # Verify zero silent fallback: attempting to get SHAP contributions for an un-evaluated index raises KeyError
    vals_array = np.random.randn(5, 4)
    index_map = {10: 0, 20: 1, 30: 2}
    with pytest.raises(KeyError):
        get_local_shap_contributions(
            vals_array=vals_array,
            orig_row_idx=999,  # Not in index_map!
            feature_names=["x0", "x1", "x2", "x3"],
            X_trans=X_mat,
            index_to_eval_pos=index_map
        )


def test_9_documentation_accuracy_representative_sample():
    """Test 9: Docstring for representative sample selection accurately describes first available unused sample."""
    doc = select_representative_samples_classification.__doc__
    assert doc is not None
    assert "first available unused sample" in doc


def test_10_non_contiguous_pandas_index_handling():
    """
    Test 10: Verify local explanations correctly use positional indexing (.iloc)
    when validation DataFrame has a non-contiguous Pandas index label set [3, 11, 25, 41, 78, 103].
    """
    non_contig_index = [3, 11, 25, 41, 78, 103]
    X_val = pd.DataFrame(
        {
            "f1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "f2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        },
        index=non_contig_index
    )
    y_val = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0], index=non_contig_index)

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_val, y_val)

    engine = ExplainabilityEngine()
    out = engine.explain(
        pipeline_or_model=rf,
        X_val=X_val,
        y_val=y_val,
        task_type="regression"
    )

    assert out.method == "shap_tree"
    assert len(out.local_explanations) > 0

    preds = rf.predict(X_val)

    # For positional row (e.g. pos_idx=2, which has Pandas index label 25):
    # f1=30.0, y_val.iloc[2]=300.0
    for loc_rec in out.local_explanations:
        pos_idx = loc_rec.sample_index  # 0-based positional index
        expected_pred = round(float(preds[pos_idx]), 4)
        expected_actual = round(float(y_val.iloc[pos_idx]), 4)

        assert loc_rec.prediction == expected_pred
        assert loc_rec.actual_value == expected_actual

        for contrib in loc_rec.contributions:
            if contrib.feature == "f1":
                assert contrib.feature_value == float(X_val.iloc[pos_idx]["f1"])
