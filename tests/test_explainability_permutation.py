"""
Unit tests for backend/explainability/permutation.py.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from backend.explainability.permutation import explain_permutation_importance


def test_explain_permutation_importance_svc():
    X_val = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6, 7, 8], "f2": [8, 7, 6, 5, 4, 3, 2, 1]})
    y_val = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    svc = SVC(random_state=42)
    svc.fit(X_val, y_val)

    res = explain_permutation_importance(svc, X_val, y_val, ["f1", "f2"], task_type="classification", n_repeats=3)
    assert len(res["global_importance"]) == 2
    assert res["global_importance"][0].rank == 1
    assert res["scoring_metric"] == "f1_macro"
    assert res["n_repeats"] == 3


def test_explain_permutation_importance_svr():
    X_val = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0], "x2": [0.1, 0.2, 0.1, 0.2, 0.1]})
    y_val = pd.Series([1.1, 2.2, 3.1, 4.2, 5.1])

    svr = SVR()
    svr.fit(X_val, y_val)

    res = explain_permutation_importance(svr, X_val, y_val, ["x1", "x2"], task_type="regression", n_repeats=3)
    assert len(res["global_importance"]) == 2
    assert res["scoring_metric"] == "neg_root_mean_squared_error"
    assert res["global_importance"][0].mean_importance is not None
