"""
Unit Tests for GENESIS-AI Week 7 Dataset Management.
"""

import pytest
import pandas as pd
from backend.evaluation.datasets import DatasetManager, DatasetManagerError


def test_dataset_manager_path_resolution():
    mgr = DatasetManager()
    path = mgr.get_dataset_path("01_numerical_classification.csv")
    assert path.exists()
    assert path.name == "01_numerical_classification.csv"


def test_dataset_manager_load():
    mgr = DatasetManager()
    df = mgr.load_dataset("01_numerical_classification.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "target" in df.columns


def test_dataset_manager_splits():
    mgr = DatasetManager()
    df = mgr.load_dataset("01_numerical_classification.csv")
    spec = mgr.get_spec("01_numerical_classification.csv")

    X_tr, X_val, X_te, y_tr, y_val, y_te = mgr.create_splits(
        df=df,
        target_column=spec.target,
        task_type=spec.task_type,
        seed=42,
        test_ratio=0.20,
        val_ratio=0.20
    )

    total_rows = len(df)
    assert len(X_tr) + len(X_val) + len(X_te) == total_rows
    assert len(y_tr) == len(X_tr)
    assert len(y_val) == len(X_val)
    assert len(y_te) == len(X_te)


def test_dataset_manager_invalid():
    mgr = DatasetManager()
    with pytest.raises(DatasetManagerError):
        mgr.get_dataset_path("non_existent_file.csv")
