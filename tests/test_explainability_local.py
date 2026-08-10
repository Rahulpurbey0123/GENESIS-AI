"""
Unit tests for backend/explainability/local.py.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from backend.explainability.local import (
    select_representative_samples_classification,
    select_representative_samples_regression,
    generate_local_explanations,
)
from backend.explainability.schemas import FeatureContribution


def test_select_representative_samples_classification():
    y_true = pd.Series([1, 0, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])

    samples = select_representative_samples_classification(y_true, y_pred, max_samples=5)
    categories = [cat for _, cat in samples]

    assert "correct_positive" in categories  # (1, 1) at idx 0
    assert "correct_negative" in categories  # (0, 0) at idx 1
    assert "false_positive" in categories    # (0, 1) at idx 2
    assert "false_negative" in categories    # (1, 0) at idx 3
    assert len(samples) <= 5


def test_select_representative_samples_regression():
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([10.1, 25.0, 30.0, 42.0, 45.0])

    samples = select_representative_samples_regression(y_true, y_pred, max_samples=5)
    categories = [cat for _, cat in samples]

    assert "low_residual" in categories
    assert "high_residual" in categories
    assert len(samples) == 5


def test_generate_local_explanations():
    X_val = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [4, 3, 2, 1]})
    y_val = pd.Series([0, 0, 1, 1])

    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_val, y_val)

    def dummy_extractor(orig_idx):
        return [FeatureContribution(feature="f1", feature_value=float(X_val.iloc[orig_idx]["f1"]), contribution=0.5)]

    recs = generate_local_explanations(
        pipeline_or_model=clf,
        X_val=X_val,
        y_val=y_val,
        X_trans=X_val.to_numpy(),
        feature_names=["f1", "f2"],
        task_type="classification",
        contribution_extractor_fn=dummy_extractor,
        max_samples=3
    )

    assert len(recs) == 3
    assert recs[0].category in ("correct_positive", "correct_negative", "false_positive", "false_negative", "representative_sample")
    assert len(recs[0].contributions) == 1
    assert recs[0].contributions[0].feature_value == float(X_val.iloc[recs[0].sample_index]["f1"])
