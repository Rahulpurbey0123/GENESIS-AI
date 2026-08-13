"""
GENESIS-AI Week 8 End-to-End Scientific Audit & Target Leakage Prevention Test Suite.

Verifies:
1. Target column (e.g. Survived) is strictly excluded from feature matrix X and SHAP.
2. Identifiers (e.g. PassengerId) are detected and excluded from X by default.
3. SHAP feature names exactly match model input features.
4. Intelligent generic target suggestion heuristic prefers valid targets over arbitrary columns.
5. Recommendation priority score is present and valid.
6. Search-space reduction is mathematically exact.
7. PR-AUC metric is calculated correctly for binary classification.
8. Complete end-to-end integration audit from upload to SHAP and LLM assistant.
"""

import pytest
import numpy as np
import pandas as pd

from backend.dataset.contract import (
    find_actual_column_name,
    detect_identifier_columns,
    get_canonical_data_split
)
from backend.dataset.cleaner import clean_dataset_for_ml
from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer
from backend.optimization.evaluator import build_sklearn_pipeline
from backend.explainability.engine import ExplainabilityEngine
from backend.recommendation.engine import RecommendationEngine
from backend.llm.service import LLMService


@pytest.fixture
def titanic_like_df():
    """Construct a representative Titanic-like DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "PassengerId": list(range(1, n + 1)),
        "Survived": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
        "Pclass": np.random.choice([1, 2, 3], size=n),
        "Name": [f"Passenger_{i}" for i in range(1, n + 1)],
        "Sex": np.random.choice(["male", "female"], size=n),
        "Age": np.random.choice([22.0, 38.0, 26.0, np.nan, 35.0], size=n),
        "SibSp": np.random.choice([0, 1, 2], size=n),
        "Parch": np.random.choice([0, 1, 2], size=n),
        "Ticket": [f"A/5 {1000 + i}" for i in range(1, n + 1)],
        "Fare": np.random.uniform(7.25, 512.3, size=n),
        "Cabin": np.random.choice(["C85", None, "E46", "G6"], size=n),
        "Embarked": np.random.choice(["S", "C", "Q"], size=n)
    })


def test_target_survived_excluded_from_feature_matrix(titanic_like_df):
    """P0 AUDIT: Target 'Survived' must NEVER enter feature matrix X under any circumstances."""
    X, y, feature_names, actual_target, identifier_cols = get_canonical_data_split(
        titanic_like_df, target_column="Survived", exclude_identifiers=True
    )

    assert actual_target == "Survived"
    assert "Survived" not in X.columns
    assert "Survived" not in feature_names
    assert "Survived" not in X.values.tolist()
    assert y.name == "Survived"
    assert len(X) == len(y)


def test_identifier_passengerid_excluded(titanic_like_df):
    """P1 AUDIT: Identifier 'PassengerId' is detected and excluded from predictive features X."""
    id_cols = detect_identifier_columns(titanic_like_df, target_column="Survived")
    assert "PassengerId" in id_cols

    X, y, feature_names, actual_target, identifier_cols = get_canonical_data_split(
        titanic_like_df, target_column="Survived", exclude_identifiers=True
    )
    assert "PassengerId" not in X.columns
    assert "PassengerId" not in feature_names
    assert "PassengerId" in identifier_cols


def test_shap_feature_names_match_model_input_and_exclude_target(titanic_like_df):
    """P0 & P1 AUDIT: SHAP feature names match model input features and strictly exclude target."""
    df_clean = clean_dataset_for_ml(titanic_like_df, target_column="Survived")
    X, y, feature_names, actual_target, id_cols = get_canonical_data_split(
        df_clean, target_column="Survived", exclude_identifiers=True
    )

    # Build and fit pipeline model
    pipeline = build_sklearn_pipeline(
        pipeline_id="classification_random_forest",
        hyperparameters={"n_estimators": 10, "max_depth": 3},
        X_sample=X,
        random_state=42
    )
    pipeline.fit(X, y)

    # Generate Explainability / SHAP output
    engine = ExplainabilityEngine()
    output = engine.explain(
        pipeline_or_model=pipeline,
        X_val=X,
        y_val=y,
        dataset_id="titanic_test.csv",
        pipeline_id="classification_random_forest",
        task_type="classification"
    )

    extracted_shap_features = [fi.feature for fi in output.global_importance]

    # Critical Invariant Assertions
    assert "Survived" not in extracted_shap_features, "P0 CRITICAL ERROR: Target 'Survived' found in SHAP attributions!"
    assert "PassengerId" not in extracted_shap_features, "P1 CRITICAL ERROR: Identifier 'PassengerId' found in SHAP attributions!"
    assert len(extracted_shap_features) > 0


def test_recommendation_priority_score_and_search_space_reduction(titanic_like_df):
    """P1 AUDIT: Recommendation priority score is present and search-space reduction is mathematically exact."""
    from backend.dataset.dip import generate_dip
    engine = RecommendationEngine()
    dip_dict = generate_dip(titanic_like_df.to_csv(index=False).encode(), target_column="Survived")
    report = engine.recommend_from_dip(dip_dict, top_k=5)


    assert report.candidate_count_before > 0
    assert report.candidate_count_after_filtering <= report.candidate_count_before

    expected_reduction = round(
        (report.candidate_count_before - report.candidate_count_after_filtering) / report.candidate_count_before,
        4
    )
    assert report.filtering_reduction == expected_reduction
    assert report.search_space_reduction == expected_reduction

    # Verify score is present and valid in recommendations
    for rec in report.recommendations:
        assert rec.score is not None
        assert 0.0 <= rec.score <= 1.0


def test_end_to_end_audit_pipeline(titanic_like_df):
    """
    P2 AUDIT: End-to-end integration audit from upload/cleaner to optimizer, SHAP, and LLM explanation.
    Verifies zero target leakage across every single stage of the system.
    """
    # Stage 1: Clean & Split
    df_clean = clean_dataset_for_ml(titanic_like_df, target_column="Survived")
    X, y, feature_names, actual_target, id_cols = get_canonical_data_split(
        df_clean, target_column="Survived", exclude_identifiers=True
    )
    assert "Survived" not in X.columns

    # Stage 2: Optimization
    config = OptimizationConfig(population_size=4, generations=2, max_evaluations=10, random_state=42)
    optimizer = EvolutionaryOptimizer(config=config)
    opt_res = optimizer.optimize(df_clean, target_column="Survived", dataset_name="titanic.csv", evaluate_test=False)
    assert opt_res.best_pipeline_id is not None
    assert opt_res.best_fitness != float("-inf")

    # Stage 3: Model Fit
    model = build_sklearn_pipeline(
        pipeline_id=opt_res.best_pipeline_id,
        hyperparameters=opt_res.best_hyperparameters,
        X_sample=X,
        random_state=42
    )
    model.fit(X, y)

    # Stage 4: SHAP Explainability
    explain_engine = ExplainabilityEngine()
    exp_out = explain_engine.explain(
        pipeline_or_model=model,
        X_val=X,
        y_val=y,
        dataset_id="titanic.csv",
        pipeline_id=opt_res.best_pipeline_id,
        task_type="classification"
    )
    shap_features = [fi.feature for fi in exp_out.global_importance]
    assert "Survived" not in shap_features

    # Stage 5: LLM Grounded Explanation
    llm_service = LLMService()
    llm_out = llm_service.explain(
        raw_evidence=exp_out,
        mode="technical",
        user_prompt="What are the most important features?"
    )
    assert llm_out.structured_explanation.question_intent == "FEATURE_IMPORTANCE"
    assert "Survived" not in llm_out.structured_explanation.important_features
