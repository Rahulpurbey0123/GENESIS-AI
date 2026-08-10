"""
Unit tests for backend/explainability/shap_explainer.py.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from backend.explainability.shap_explainer import explain_shap_tree, get_local_shap_contributions, HAS_SHAP


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP is not installed")
def test_explain_shap_tree_classification():
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 2, size=50)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    feature_names = ["f1", "f2", "f3", "f4"]
    res = explain_shap_tree(rf, X, feature_names)

    assert "global_importance" in res
    assert len(res["global_importance"]) == 4
    assert res["global_importance"][0].rank == 1
    assert res["base_value"] is not None
    assert res["eval_count"] == 50


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP is not installed")
def test_explain_shap_tree_regression():
    X = np.random.rand(50, 3)
    y = np.random.rand(50) * 10.0
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)

    feature_names = ["a", "b", "c"]
    res = explain_shap_tree(rf, X, feature_names)

    assert len(res["global_importance"]) == 3
    assert res["global_importance"][0].rank == 1


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP is not installed")
def test_get_local_shap_contributions():
    vals_array = np.array([[0.5, -0.2, 0.1], [0.3, 0.4, -0.5]])
    feature_names = ["x1", "x2", "x3"]
    X_trans = np.random.rand(2, 3)

    contribs = get_local_shap_contributions(
        vals_array=vals_array,
        orig_row_idx=0,
        feature_names=feature_names,
        X_trans=X_trans
    )

    assert len(contribs) == 3
    # Should be sorted by absolute magnitude descending: 0.5 (x1), |-0.2| (x2), 0.1 (x3)
    assert contribs[0].feature == "x1"
    assert contribs[0].contribution == 0.5
