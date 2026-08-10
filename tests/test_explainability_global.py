"""
Unit tests for backend/explainability/global.py.
"""

import pytest
from backend.explainability.schemas import FeatureImportanceRecord
import importlib
_global_mod = importlib.import_module("backend.explainability.global")
format_global_importance = _global_mod.format_global_importance


def test_format_global_importance_ranking():
    recs = [
        FeatureImportanceRecord(feature="feat_b", importance=0.3, rank=2),
        FeatureImportanceRecord(feature="feat_a", importance=0.7, rank=1),
    ]

    formatted = format_global_importance(recs)
    assert len(formatted) == 2
    assert formatted[0].feature == "feat_a"
    assert formatted[0].rank == 1
    assert formatted[1].feature == "feat_b"
    assert formatted[1].rank == 2


def test_format_global_importance_empty():
    assert format_global_importance([]) == []
