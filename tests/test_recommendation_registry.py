"""
Tests for Candidate Pipeline Registry (backend/recommendation/registry.py).
"""

import pytest
from backend.recommendation.registry import (
    PipelineRegistry,
    CLASSIFICATION_PIPELINES,
    REGRESSION_PIPELINES,
)


def test_registry_pipeline_counts():
    """Verify total count and category counts in central pipeline registry."""
    registry = PipelineRegistry()
    assert registry.total_count() == 10
    assert len(CLASSIFICATION_PIPELINES) == 5
    assert len(REGRESSION_PIPELINES) == 5


def test_registry_get_by_task():
    """Verify registry filtering by task type."""
    registry = PipelineRegistry()
    clf_pipelines = registry.get_pipelines_by_task("classification")
    reg_pipelines = registry.get_pipelines_by_task("regression")

    assert len(clf_pipelines) == 5
    assert all(p.task == "classification" for p in clf_pipelines)

    assert len(reg_pipelines) == 5
    assert all(p.task == "regression" for p in reg_pipelines)


def test_registry_get_by_id():
    """Verify lookup of specific pipelines by ID."""
    registry = PipelineRegistry()

    rf_clf = registry.get_pipeline_by_id("classification_random_forest")
    assert rf_clf is not None
    assert rf_clf.model_name == "RandomForestClassifier"
    assert rf_clf.task == "classification"
    assert rf_clf.model_family == "tree_ensemble"

    linear_reg = registry.get_pipeline_by_id("regression_linear_regression")
    assert linear_reg is not None
    assert linear_reg.model_name == "LinearRegression"
    assert linear_reg.task == "regression"

    non_existent = registry.get_pipeline_by_id("invalid_id")
    assert non_existent is None


def test_pipeline_metadata_fields():
    """Verify that all pipeline metadata fields are populated correctly."""
    registry = PipelineRegistry()
    for pipeline in registry.get_all_pipelines():
        assert isinstance(pipeline.pipeline_id, str)
        assert isinstance(pipeline.name, str)
        assert pipeline.task in ("classification", "regression")
        assert pipeline.model_family in ("linear", "tree_ensemble", "svm", "knn")
        assert isinstance(pipeline.steps, list)
        assert len(pipeline.steps) >= 2  # At least imputer and model
        assert isinstance(pipeline.requires_scaling, bool)
        assert isinstance(pipeline.supports_class_weight, bool)
        assert isinstance(pipeline.handles_categorical_natively, bool)
        assert pipeline.computational_cost in ("low", "medium", "high")
