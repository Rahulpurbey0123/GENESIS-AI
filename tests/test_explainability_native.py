"""
Unit tests for backend/explainability/native_importance.py.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from backend.explainability.native_importance import (
    explain_linear_coefficients,
    get_local_linear_contributions,
    explain_native_tree_importance
)


def test_explain_linear_coefficients_logistic():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    y = np.array([0, 0, 1, 1])

    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)

    res = explain_linear_coefficients(lr, X, ["feat_a", "feat_b"])
    assert len(res["global_importance"]) == 2
    assert res["global_importance"][0].rank == 1
    assert res["global_importance"][0].direction in (1, -1)


def test_explain_linear_coefficients_regression():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y = np.array([2.0, 3.5, 5.0, 6.5])

    reg = LinearRegression()
    reg.fit(X, y)

    res = explain_linear_coefficients(reg, X, ["x1", "x2"])
    assert len(res["global_importance"]) == 2
    assert res["intercept"] is not None


def test_explain_native_tree_importance():
    X = np.random.rand(40, 3)
    y = np.random.randint(0, 2, size=40)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    res = explain_native_tree_importance(rf, ["f1", "f2", "f3"])
    assert len(res["global_importance"]) == 3
    assert res["global_importance"][0].rank == 1
    assert res["global_importance"][0].direction is None


def test_get_local_linear_contributions():
    weights = np.array([2.0, -1.5])
    sample_row = np.array([3.0, 4.0])

    contribs = get_local_linear_contributions(weights, sample_row, ["a", "b"])
    # 2.0 * 3.0 = 6.0; -1.5 * 4.0 = -6.0
    assert len(contribs) == 2
    assert abs(contribs[0].contribution) == 6.0
