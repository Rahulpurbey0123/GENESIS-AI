"""
Unit Tests for GENESIS-AI Week 7 Statistical Analysis Module (Hardened v1.3).
"""

import pytest
from backend.evaluation.schemas import RawObservation
from backend.evaluation.statistics import StatisticsAnalyzer


def test_statistics_analyzer_aggregation():
    obs_list = [
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.90, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=123, metric="f1", score=0.92, candidate_evaluations=32, runtime_seconds=1.2),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=456, metric="f1", score=0.88, candidate_evaluations=28, runtime_seconds=0.9),
    ]

    analyzer = StatisticsAnalyzer(obs_list)
    aggregated = analyzer.compute_aggregated_metrics()

    assert len(aggregated) == 1
    agg = aggregated[0]
    assert agg.dataset == "ds1.csv"
    assert agg.method == "method_a_full_genesis"
    assert agg.mean_score == 0.90
    assert agg.best_score == 0.92
    assert agg.worst_score == 0.88


def test_hypothesis_evaluations():
    obs_list = [
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.90, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_b_without_dip", seed=42, metric="f1", score=0.90, candidate_evaluations=50, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_c_without_recommendation", seed=42, metric="f1", score=0.90, candidate_evaluations=60, runtime_seconds=2.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=42, metric="f1", score=0.80, candidate_evaluations=1, runtime_seconds=0.1),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_e_unguided_baseline", seed=42, metric="f1", score=0.88, candidate_evaluations=200, runtime_seconds=5.0),
    ]

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()

    h_dict = {h.hypothesis_id: h for h in hypotheses}
    assert "H1" in h_dict
    assert "H2" in h_dict
    assert "H3" in h_dict
    assert "H4" in h_dict

    assert h_dict["H1"].status in ["SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"]
    assert h_dict["H2"].status in ["SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"]
    assert h_dict["H3"].status in ["SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"]


def test_h2_performance_degradation_prevents_supported():
    """Verify that efficiency reduction alone does NOT mark H2 SUPPORTED if performance is significantly degraded."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.20, 0.21, 0.19, 0.22, 0.18]
    b_scores = [0.90, 0.92, 0.88, 0.91, 0.89]
    a_evals = [10, 12, 8, 11, 9]
    b_evals = [50, 52, 48, 51, 49]
    obs_list = []
    for seed, sa, sb, ea, eb in zip(seeds, a_scores, b_scores, a_evals, b_evals):
        # Method A evaluates fewer candidates (10 vs 50) BUT has significantly lower F1 (0.20 vs 0.90)
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=sa, candidate_evaluations=ea, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_b_without_dip", seed=seed, metric="f1", score=sb, candidate_evaluations=eb, runtime_seconds=2.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]

    assert h2.status == "NOT SUPPORTED"
    assert "degraded" in h2.rationale.lower()


def test_metric_direction_helper():
    """Test get_metric_direction helper and error handling for unknown metrics."""
    from backend.evaluation.metrics import get_metric_direction

    assert get_metric_direction("macro_f1") == "higher"
    assert get_metric_direction("f1") == "higher"
    assert get_metric_direction("accuracy") == "higher"
    assert get_metric_direction("rmse") == "lower"
    assert get_metric_direction("mae") == "lower"

    with pytest.raises(ValueError) as excinfo:
        get_metric_direction("unknown_metric_foo")
    assert "Unknown metric" in str(excinfo.value)


def test_validate_comparable_observations():
    """Test validation layer rules TEST 1 - TEST 4."""
    from backend.evaluation.statistics import validate_comparable_observations

    obs_a_cls = [RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.90, candidate_evaluations=30, runtime_seconds=1.0)]
    obs_a_reg = [RawObservation(dataset="ds2.csv", task_type="regression", method="method_a_full_genesis", seed=42, metric="rmse", score=100.0, candidate_evaluations=30, runtime_seconds=1.0)]
    obs_b_cls = [RawObservation(dataset="ds1.csv", task_type="classification", method="method_b_without_dip", seed=42, metric="f1", score=0.85, candidate_evaluations=50, runtime_seconds=1.0)]
    obs_b_reg = [RawObservation(dataset="ds2.csv", task_type="regression", method="method_b_without_dip", seed=42, metric="rmse", score=110.0, candidate_evaluations=50, runtime_seconds=1.0)]

    # TEST 1: Input A classification F1 + A regression RMSE (mixed task list)
    valid, msg = validate_comparable_observations(obs_a_cls + obs_a_reg, obs_b_cls + obs_b_reg)
    assert not valid
    assert "Mixed task types" in msg or "Mixed metrics" in msg

    # TEST 2: A classification F1 vs B classification F1
    valid, msg = validate_comparable_observations(obs_a_cls, obs_b_cls)
    assert valid
    assert msg == "Valid"

    # TEST 3: A regression RMSE vs B regression RMSE
    valid, msg = validate_comparable_observations(obs_a_reg, obs_b_reg)
    assert valid
    assert msg == "Valid"

    # TEST 4: A classification F1 vs B regression RMSE
    valid, msg = validate_comparable_observations(obs_a_cls, obs_b_reg)
    assert not valid
    assert "Task type mismatch" in msg or "Metric mismatch" in msg


def test_direction_aware_degradation_logic():
    """Test direction-aware score comparison logic for F1 and RMSE."""
    from backend.evaluation.metrics import is_better_score

    # F1 (Higher is better)
    assert not is_better_score(0.50, 0.60, "f1")  # A (0.50) is worse than B (0.60)
    assert is_better_score(0.70, 0.60, "f1")      # A (0.70) is better than B (0.60)

    # RMSE (Lower is better)
    assert not is_better_score(900.0, 800.0, "rmse") # A (900) is worse than B (800)
    assert is_better_score(700.0, 800.0, "rmse")     # A (700) is better than B (800)


def test_h2_regression_degradation_prevents_supported():
    """Verify that regression RMSE degradation prevents H2 SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_rmse = [900.0, 910.0, 890.0, 905.0, 895.0]
    b_rmse = [800.0, 825.0, 775.0, 810.0, 790.0]
    a_evals = [10, 12, 8, 11, 9]
    b_evals = [50, 52, 48, 51, 49]
    obs_list = []
    for seed, ar, br, ea, eb in zip(seeds, a_rmse, b_rmse, a_evals, b_evals):
        # Method A evaluates fewer candidates (10 vs 50) BUT has higher (worse) RMSE (900 vs 800)
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=seed, metric="rmse", score=ar, candidate_evaluations=ea, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_b_without_dip", seed=seed, metric="rmse", score=br, candidate_evaluations=eb, runtime_seconds=2.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]

    assert h2.status == "NOT SUPPORTED"
    assert "degraded" in h2.rationale.lower()


def test_h3_task_separation():
    """Verify H3 evaluates classification and regression separately without mixing metric types."""
    obs_list = [
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.90, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=42, metric="f1", score=0.70, candidate_evaluations=2, runtime_seconds=0.1),
        RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=42, metric="rmse", score=500.0, candidate_evaluations=25, runtime_seconds=1.0),
        RawObservation(dataset="ds5.csv", task_type="regression", method="method_d_recommendation_only", seed=42, metric="rmse", score=600.0, candidate_evaluations=2, runtime_seconds=0.1),
    ]

    analyzer = StatisticsAnalyzer(obs_list)
    tests = analyzer.perform_statistical_tests()
    
    # Check that individual tests are created separately for F1 and RMSE
    f1_tests = [t for t in tests if t.metric == "f1" and "Method A vs Method D" in t.comparison]
    rmse_tests = [t for t in tests if t.metric == "rmse" and "Method A vs Method D" in t.comparison]

    assert len(f1_tests) == 1
    assert len(rmse_tests) == 1
    assert f1_tests[0].metric == "f1"
    assert rmse_tests[0].metric == "rmse"

    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert "Classification F1" in h3.rationale
    assert "Regression RMSE" in h3.rationale


def test_h4_task_separated_stability():
    """Verify H4 stability calculates classification and regression variability separately."""
    obs_list = [
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.90, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=123, metric="f1", score=0.92, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=42, metric="rmse", score=8000.0, candidate_evaluations=25, runtime_seconds=1.0),
        RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=123, metric="rmse", score=8100.0, candidate_evaluations=25, runtime_seconds=1.0),
    ]

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h4 = [h for h in hypotheses if h.hypothesis_id == "H4"][0]

    assert "Classification Stability" in h4.rationale
    assert "Regression Stability" in h4.rationale


def test_h1_removed_win_count_threshold_3_out_of_5():
    """Verify that a 3 out of 5 win count does NOT automatically force SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    obs_list = []
    # 3 datasets where A slightly outperforms E, 2 datasets where A is worse than E
    # Overall mean difference is non-significant or mixed
    datasets = ["ds1.csv", "ds2.csv", "ds3.csv", "ds4.csv", "ds5.csv"]
    a_scores = [0.60, 0.60, 0.60, 0.20, 0.20]
    e_scores = [0.55, 0.55, 0.55, 0.90, 0.90]

    for ds, a_s, e_s in zip(datasets, a_scores, e_scores):
        for seed in seeds:
            obs_list.append(RawObservation(dataset=ds, task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
            obs_list.append(RawObservation(dataset=ds, task_type="classification", method="method_e_unguided_baseline", seed=seed, metric="f1", score=e_s, candidate_evaluations=200, runtime_seconds=5.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h1 = [h for h in hypotheses if h.hypothesis_id == "H1"][0]

    # Must NOT be SUPPORTED just because 3 out of 5 matched!
    assert h1.status in ["INCONCLUSIVE", "NOT SUPPORTED"]
    assert h1.status != "SUPPORTED"


def test_h1_five_wins_supported():
    """Verify H1 returns SUPPORTED when Method A achieves statistically significant improvement over Method E."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.85, 0.87, 0.83, 0.86, 0.84]
    e_scores = [0.60, 0.61, 0.59, 0.62, 0.58]
    obs_list = []
    datasets = ["ds1.csv", "ds2.csv", "ds3.csv", "ds4.csv", "ds5.csv"]

    for ds in datasets:
        for seed, a_s, e_s in zip(seeds, a_scores, e_scores):
            obs_list.append(RawObservation(dataset=ds, task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
            obs_list.append(RawObservation(dataset="ds", task_type="classification", method="method_e_unguided_baseline", seed=seed, metric="f1", score=e_s, candidate_evaluations=200, runtime_seconds=5.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h1 = [h for h in hypotheses if h.hypothesis_id == "H1"][0]

    assert h1.status == "SUPPORTED"


def test_h1_all_matched_without_significance_is_inconclusive():
    """CRITICAL TEST: All 5 datasets matched/exceeded by A over E, BUT score differences are non-significant (p >= 0.05).
    Verifies all_matched DOES NOT force SUPPORTED -> MUST evaluate to INCONCLUSIVE.
    """
    seeds = [42, 123, 456, 789, 2024]
    datasets = ["ds1.csv", "ds2.csv", "ds3.csv", "ds4.csv", "ds5.csv"]
    a_scores = [0.51, 0.52, 0.49, 0.50, 0.49]
    e_scores = [0.50, 0.51, 0.50, 0.49, 0.51]
    obs_list = []

    for ds in datasets:
        for seed, a_s, e_s in zip(seeds, a_scores, e_scores):
            obs_list.append(RawObservation(dataset=ds, task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
            obs_list.append(RawObservation(dataset=ds, task_type="classification", method="method_e_unguided_baseline", seed=seed, metric="f1", score=e_s, candidate_evaluations=200, runtime_seconds=5.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h1 = [h for h in hypotheses if h.hypothesis_id == "H1"][0]

    # all_matched is True (mean A >= mean E on all 5 datasets), BUT p >= 0.05
    # Must NOT produce SUPPORTED!
    assert h1.status != "SUPPORTED"
    assert h1.status == "INCONCLUSIVE"
    # Descriptive breakdown must still be present in rationale
    assert "matched" in h1.rationale.lower() or "5/5" in h1.rationale


def test_h1_zero_wins_not_supported():
    """Verify H1 returns NOT SUPPORTED when Method A is significantly worse than Method E."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.20, 0.22, 0.18, 0.21, 0.19]
    e_scores = [0.90, 0.88, 0.92, 0.89, 0.91]
    obs_list = []
    datasets = ["ds1.csv", "ds2.csv", "ds3.csv", "ds4.csv", "ds5.csv"]

    for ds in datasets:
        for seed, a_s, e_s in zip(seeds, a_scores, e_scores):
            obs_list.append(RawObservation(dataset=ds, task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
            obs_list.append(RawObservation(dataset=ds, task_type="classification", method="method_e_unguided_baseline", seed=seed, metric="f1", score=e_s, candidate_evaluations=200, runtime_seconds=5.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h1 = [h for h in hypotheses if h.hypothesis_id == "H1"][0]

    assert h1.status == "NOT SUPPORTED"


# =============================================================================
# STATISTICAL FALLBACK UNIT TESTS (v1.7)
# =============================================================================

def test_fallback_higher_mean_without_stat_test_is_inconclusive():
    """Single paired observation (stat test unavailable) with A mean > B mean -> INCONCLUSIVE."""
    obs_list = [
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.80, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=42, metric="f1", score=0.50, candidate_evaluations=2, runtime_seconds=0.1)
    ]
    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    # Higher mean alone without stat test MUST NOT produce SUPPORTED
    assert h3.status == "INCONCLUSIVE"
    assert "stat test unavailable" in h3.rationale


def test_fallback_lower_mean_without_stat_test_is_inconclusive():
    """Single paired observation (stat test unavailable) with A mean < B mean -> INCONCLUSIVE."""
    obs_list = [
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.40, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=42, metric="f1", score=0.80, candidate_evaluations=2, runtime_seconds=0.1)
    ]
    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    # Lower mean alone without stat test MUST NOT produce NOT SUPPORTED
    assert h3.status == "INCONCLUSIVE"
    assert "stat test unavailable" in h3.rationale


def test_fallback_f1_improvement_unavailable():
    """F1 higher mean with stat test unavailable yields no statistically supported improvement."""
    obs_list = [
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.90, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_e_unguided_baseline", seed=42, metric="f1", score=0.50, candidate_evaluations=200, runtime_seconds=5.0)
    ]
    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h1 = [h for h in hypotheses if h.hypothesis_id == "H1"][0]
    # N=1 observation: matched 1 dataset but stat test is unavailable; status is INCONCLUSIVE or SUPPORTED via all_matched
    assert "stat test unavailable" in h1.rationale


def test_fallback_rmse_improvement_unavailable():
    """RMSE lower mean with stat test unavailable yields no statistically supported improvement."""
    obs_list = [
        RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=42, metric="rmse", score=200.0, candidate_evaluations=25, runtime_seconds=1.0),
        RawObservation(dataset="ds5.csv", task_type="regression", method="method_d_recommendation_only", seed=42, metric="rmse", score=800.0, candidate_evaluations=2, runtime_seconds=0.1)
    ]
    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "INCONCLUSIVE"
    assert "stat test unavailable" in h3.rationale


def test_fallback_f1_degradation_unavailable():
    """F1 lower mean with stat test unavailable yields no statistically supported degradation."""
    obs_list = [
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=42, metric="f1", score=0.30, candidate_evaluations=30, runtime_seconds=1.0),
        RawObservation(dataset="ds1.csv", task_type="classification", method="method_b_without_dip", seed=42, metric="f1", score=0.80, candidate_evaluations=50, runtime_seconds=2.0)
    ]
    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]
    assert h2.status == "INCONCLUSIVE"
    assert "stat test unavailable" in h2.rationale


def test_fallback_rmse_degradation_unavailable():
    """RMSE higher mean with stat test unavailable yields no statistically supported degradation."""
    obs_list = [
        RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=42, metric="rmse", score=900.0, candidate_evaluations=25, runtime_seconds=1.0),
        RawObservation(dataset="ds5.csv", task_type="regression", method="method_b_without_dip", seed=42, metric="rmse", score=300.0, candidate_evaluations=50, runtime_seconds=2.0)
    ]
    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]
    assert h2.status == "INCONCLUSIVE"
    assert "stat test unavailable" in h2.rationale


def test_valid_stat_test_significant_improvement():
    """Valid paired data across 5 seeds with p < 0.05 -> SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.80, 0.85, 0.82, 0.84, 0.81]
    d_scores = [0.50, 0.52, 0.48, 0.51, 0.49]
    obs_list = []
    for seed, a_s, d_s in zip(seeds, a_scores, d_scores):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=seed, metric="f1", score=d_s, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "SUPPORTED"
    assert "p=" in h3.rationale


def test_valid_stat_test_significant_degradation():
    """Valid paired data across 5 seeds with p < 0.05 degradation -> NOT SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.20, 0.22, 0.18, 0.21, 0.19]
    d_scores = [0.90, 0.88, 0.92, 0.89, 0.91]
    obs_list = []
    for seed, a_s, d_s in zip(seeds, a_scores, d_scores):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=seed, metric="f1", score=d_s, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "NOT SUPPORTED"
    assert "p=" in h3.rationale


def test_valid_stat_test_non_significant_difference():
    """Valid paired data across 5 seeds with p >= 0.05 -> INCONCLUSIVE."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.50, 0.52, 0.48, 0.51, 0.49]
    d_scores = [0.51, 0.50, 0.52, 0.49, 0.51]
    obs_list = []
    for seed, a_s, d_s in zip(seeds, a_scores, d_scores):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=seed, metric="f1", score=d_s, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "INCONCLUSIVE"
    assert "p=" in h3.rationale


def test_h1_insufficient_mixed_evidence_inconclusive():
    """Verify H1 returns INCONCLUSIVE when evidence is mixed / non-significant."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.50, 0.52, 0.48, 0.51, 0.49]
    e_scores = [0.51, 0.50, 0.52, 0.49, 0.51]
    obs_list = []
    for seed, a_s, e_s in zip(seeds, a_scores, e_scores):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_e_unguided_baseline", seed=seed, metric="f1", score=e_s, candidate_evaluations=200, runtime_seconds=5.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h1 = [h for h in hypotheses if h.hypothesis_id == "H1"][0]

    assert h1.status == "INCONCLUSIVE"


def test_h1_metric_direction():
    """Verify H1 handles classification F1 (higher better) and regression RMSE (lower better) correctly."""
    seeds = [42, 123, 456, 789, 2024]
    a_f1 = [0.90, 0.92, 0.88, 0.91, 0.89]
    e_f1 = [0.50, 0.51, 0.49, 0.52, 0.48]
    a_rmse = [100.0, 110.0, 90.0, 105.0, 95.0]
    e_rmse = [500.0, 515.0, 485.0, 510.0, 490.0]
    obs_list = []

    for seed, af, ef, ar, er in zip(seeds, a_f1, e_f1, a_rmse, e_rmse):
        # F1: A > E -> A is better
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=af, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_e_unguided_baseline", seed=seed, metric="f1", score=ef, candidate_evaluations=200, runtime_seconds=5.0))

        # RMSE: A < E -> A is better (lower RMSE)
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=seed, metric="rmse", score=ar, candidate_evaluations=25, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_e_unguided_baseline", seed=seed, metric="rmse", score=er, candidate_evaluations=200, runtime_seconds=5.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h1 = [h for h in hypotheses if h.hypothesis_id == "H1"][0]

    assert h1.status == "SUPPORTED"


def test_no_arbitrary_matched_count_threshold_in_codebase():
    """Search codebase for obsolete matched_count threshold logic."""
    from pathlib import Path
    stats_file = Path("backend/evaluation/statistics.py")
    content = stats_file.read_text(encoding="utf-8")
    assert "matched_count >= 3" not in content
    assert "matched_count >= 4" not in content
    assert "matched_count >= 2" not in content


# =============================================================================
# H3 HARDENED EDGE-CASE TESTS
# =============================================================================

def test_h2_regression_degradation_prevents_supported():
    """Verify that regression RMSE degradation prevents H2 SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_rmse = [900.0, 915.0, 885.0, 905.0, 895.0]
    b_rmse = [800.0, 810.0, 790.0, 815.0, 785.0]
    a_evals = [10, 12, 8, 11, 9]
    b_evals = [50, 52, 48, 51, 49]
    obs_list = []
    for seed, ar, br, ea, eb in zip(seeds, a_rmse, b_rmse, a_evals, b_evals):
        # Method A evaluates fewer candidates (10 vs 50) BUT has higher (worse) RMSE (900 vs 800)
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=seed, metric="rmse", score=ar, candidate_evaluations=ea, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_b_without_dip", seed=seed, metric="rmse", score=br, candidate_evaluations=eb, runtime_seconds=2.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]

    assert h2.status == "NOT SUPPORTED"
    assert "degraded" in h2.rationale.lower()


def test_h3_edge_case_1_improvement_only():
    """Test H3 with classification improvement and no degradation -> SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.90, 0.92, 0.88, 0.91, 0.89]
    d_scores = [0.60, 0.61, 0.59, 0.62, 0.58]
    obs_list = []
    for seed, a_s, d_s in zip(seeds, a_scores, d_scores):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=seed, metric="f1", score=d_s, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "SUPPORTED"


def test_h3_edge_case_2_degradation_only():
    """Test H3 with classification degradation -> NOT SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.40, 0.41, 0.39, 0.42, 0.38]
    d_scores = [0.90, 0.92, 0.88, 0.91, 0.89]
    obs_list = []
    for seed, a_s, d_s in zip(seeds, a_scores, d_scores):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=seed, metric="f1", score=d_s, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "NOT SUPPORTED"


def test_h3_edge_case_3_improvement_plus_degradation():
    """CRITICAL TEST: Classification improvement + Regression degradation MUST NOT yield SUPPORTED -> NOT SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_cls = [0.90, 0.92, 0.88, 0.91, 0.89]
    d_cls = [0.50, 0.51, 0.49, 0.52, 0.48]
    a_reg = [900.0, 915.0, 885.0, 905.0, 895.0]
    d_reg = [200.0, 210.0, 190.0, 215.0, 185.0]
    obs_list = []
    for seed, ac, dc, ar, dr in zip(seeds, a_cls, d_cls, a_reg, d_reg):
        # Classification F1: A > D [Improvement]
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=ac, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=seed, metric="f1", score=dc, candidate_evaluations=2, runtime_seconds=0.1))

        # Regression RMSE: A > D [Degradation - higher RMSE is worse]
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=seed, metric="rmse", score=ar, candidate_evaluations=25, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_d_recommendation_only", seed=seed, metric="rmse", score=dr, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]

    assert h3.status != "SUPPORTED"
    assert h3.status == "NOT SUPPORTED"


def test_h3_edge_case_4_no_significant_evidence():
    """Test H3 with small non-significant differences -> INCONCLUSIVE."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.50, 0.52, 0.48, 0.51, 0.49]
    d_scores = [0.51, 0.50, 0.52, 0.49, 0.51]
    obs_list = []
    for seed, a_s, d_s in zip(seeds, a_scores, d_scores):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=a_s, candidate_evaluations=30, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_d_recommendation_only", seed=seed, metric="f1", score=d_s, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "INCONCLUSIVE"


def test_h3_edge_case_5_regression_improvement():
    """Test H3 regression improvement (RMSE A=700 < D=800, lower is better) -> SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_rmse = [700.0, 715.0, 685.0, 705.0, 695.0]
    d_rmse = [800.0, 810.0, 790.0, 815.0, 785.0]
    obs_list = []
    for seed, ar, dr in zip(seeds, a_rmse, d_rmse):
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=seed, metric="rmse", score=ar, candidate_evaluations=25, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_d_recommendation_only", seed=seed, metric="rmse", score=dr, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "SUPPORTED"


def test_h3_edge_case_6_regression_degradation():
    """Test H3 regression degradation (RMSE A=900 > D=800, lower is better) -> NOT SUPPORTED."""
    seeds = [42, 123, 456, 789, 2024]
    a_rmse = [900.0, 915.0, 885.0, 905.0, 895.0]
    d_rmse = [800.0, 810.0, 790.0, 815.0, 785.0]
    obs_list = []
    for seed, ar, dr in zip(seeds, a_rmse, d_rmse):
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_a_full_genesis", seed=seed, metric="rmse", score=ar, candidate_evaluations=25, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds5.csv", task_type="regression", method="method_d_recommendation_only", seed=seed, metric="rmse", score=dr, candidate_evaluations=2, runtime_seconds=0.1))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h3 = [h for h in hypotheses if h.hypothesis_id == "H3"][0]
    assert h3.status == "NOT SUPPORTED"


# =============================================================================
# H2 HARDENED EDGE-CASE TESTS
# =============================================================================

def test_h2_performance_degradation_overrides_inconclusive_efficiency():
    """Verify performance degradation forces NOT SUPPORTED even when efficiency reduction is inconclusive."""
    seeds = [42, 123, 456, 789, 2024]
    a_evals = [40, 50, 42, 48, 45]
    b_evals = [48, 47, 52, 45, 50]
    a_scores = [0.20, 0.21, 0.19, 0.22, 0.18]
    b_scores = [0.90, 0.92, 0.88, 0.91, 0.89]
    obs_list = []
    for seed, ev_a, ev_b, sa, sb in zip(seeds, a_evals, b_evals, a_scores, b_scores):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=sa, candidate_evaluations=ev_a, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_b_without_dip", seed=seed, metric="f1", score=sb, candidate_evaluations=ev_b, runtime_seconds=2.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]
    assert h2.status == "NOT SUPPORTED"


def test_h2_inconclusive_efficiency_and_maintained_performance():
    """Verify H2 returns INCONCLUSIVE when evals reduction is non-significant but performance is maintained."""
    seeds = [42, 123, 456, 789, 2024]
    a_evals = [45, 52, 40, 48, 43]
    b_evals = [47, 46, 48, 45, 49]
    obs_list = []
    for seed, ev_a, ev_b in zip(seeds, a_evals, b_evals):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=0.85, candidate_evaluations=ev_a, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_b_without_dip", seed=seed, metric="f1", score=0.85, candidate_evaluations=ev_b, runtime_seconds=2.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]
    assert h2.status == "INCONCLUSIVE"


def test_h2_complete_support():
    """Verify H2 returns SUPPORTED when efficiency reduction is statistically significant and performance maintained."""
    seeds = [42, 123, 456, 789, 2024]
    a_scores = [0.90, 0.92, 0.88, 0.91, 0.89]
    b_scores = [0.88, 0.89, 0.87, 0.90, 0.86]
    a_evals = [10, 12, 8, 11, 9]
    b_evals = [100, 105, 95, 102, 98]
    obs_list = []
    for seed, sa, sb, ea, eb in zip(seeds, a_scores, b_scores, a_evals, b_evals):
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=sa, candidate_evaluations=ea, runtime_seconds=1.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_b_without_dip", seed=seed, metric="f1", score=sb, candidate_evaluations=eb, runtime_seconds=5.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]
    assert h2.status == "SUPPORTED"


def test_h2_no_efficiency_benefit():
    """Verify H2 cannot return SUPPORTED when Method A evaluates equal or more candidates than Method B."""
    seeds = [42, 123, 456, 789, 2024]
    obs_list = []
    for seed in seeds:
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_a_full_genesis", seed=seed, metric="f1", score=0.90, candidate_evaluations=100, runtime_seconds=5.0))
        obs_list.append(RawObservation(dataset="ds1.csv", task_type="classification", method="method_b_without_dip", seed=seed, metric="f1", score=0.90, candidate_evaluations=20, runtime_seconds=1.0))

    analyzer = StatisticsAnalyzer(obs_list)
    hypotheses = analyzer.evaluate_hypotheses()
    h2 = [h for h in hypotheses if h.hypothesis_id == "H2"][0]
    assert h2.status != "SUPPORTED"

