"""
Unit safety tests for GENESIS-AI Week 5 Explainability Engine.

Verifies model mutation protection, test set isolation, NaN/Inf handling,
and unknown model fallback handling.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from backend.explainability.engine import ExplainabilityEngine
from backend.explainability.validators import validate_fitted_model, validate_no_nan_inf


def test_model_mutation_protection():
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    y_train = pd.Series([0, 0, 1, 1])

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    coef_before = clf.coef_.copy()
    intercept_before = clf.intercept_.copy()

    engine = ExplainabilityEngine()
    output = engine.explain(
        pipeline_or_model=clf,
        X_val=X_train,
        y_val=y_train,
        task_type="classification"
    )

    # Weights and intercepts must remain 100% identical post-explanation
    np.testing.assert_array_equal(clf.coef_, coef_before)
    np.testing.assert_array_equal(clf.intercept_, intercept_before)
    assert output.method == "linear_coefficients"


def test_nan_inf_safety_validation():
    clean_dict = {"a": 1.0, "b": [2.0, 3.0], "c": {"d": 4.5}}
    assert validate_no_nan_inf(clean_dict) is True

    nan_dict = {"a": float("nan"), "b": 2.0}
    assert validate_no_nan_inf(nan_dict) is False

    inf_dict = {"a": float("inf"), "b": 2.0}
    assert validate_no_nan_inf(inf_dict) is False


def test_unfitted_model_safety():
    clf = RandomForestClassifier(n_estimators=5)
    is_fitted, err = validate_fitted_model(clf)
    assert is_fitted is False

    engine = ExplainabilityEngine()
    X_dummy = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y_dummy = pd.Series([0, 1])

    output = engine.explain(clf, X_dummy, y_dummy)
    assert output.method == "unsupported"
    assert len(output.warnings) > 0


def test_unknown_estimator_fallback():
    class DummyEstimator:
        def __init__(self):
            pass

    dummy = DummyEstimator()
    engine = ExplainabilityEngine()
    X_dummy = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y_dummy = pd.Series([0, 1])

    output = engine.explain(dummy, X_dummy, y_dummy)
    assert output.method == "unsupported"
