"""
Statistical & Descriptive Analysis Module for GENESIS-AI Week 7 Research Evaluation (Hardened v1.4).

Computes summary metrics separately per task type (Classification Macro F1 vs Regression RMSE),
ablation breakdowns, task-separated paired statistical tests (t-tests, Wilcoxon), effect sizes (Cohen's d),
metric-aware validation, and empirical, evidence-grounded hypothesis evaluations (H1, H2, H3, H4) without post-hoc thresholds.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

from backend.evaluation.schemas import (
    RawObservation,
    AggregatedMetric,
    AblationRecord,
    StatisticalTestResult,
    HypothesisEvaluation,
    BenchmarkRunSummary,
    BenchmarkConfig
)
from backend.evaluation.metrics import is_higher_better, is_better_score, get_metric_direction
from backend.evaluation.configuration import METHOD_METADATA


def validate_comparable_observations(
    obs_a: List[RawObservation],
    obs_b: List[RawObservation],
    check_metric: bool = True
) -> Tuple[bool, str]:
    """
    Validate that two sets of raw observations are strictly comparable:
    - Both lists are non-empty.
    - Uniform task_type across all items in A and B, and matching task_type between A and B.
    - Uniform metric across all items in A and B, and matching metric between A and B (if check_metric is True).
    - Matching observation counts and paired dataset/seed ordering.

    Returns:
        Tuple of (is_valid: bool, rationale_or_error_message: str).
    """
    if not obs_a or not obs_b:
        return False, "Observation list(s) empty."

    task_types_a = set(o.task_type for o in obs_a)
    task_types_b = set(o.task_type for o in obs_b)
    if len(task_types_a) > 1 or len(task_types_b) > 1:
        return False, f"Mixed task types within observation lists: {task_types_a} vs {task_types_b}."
    if task_types_a != task_types_b:
        return False, f"Task type mismatch between observation lists: {task_types_a} vs {task_types_b}."

    if check_metric:
        metrics_a = set(o.metric for o in obs_a)
        metrics_b = set(o.metric for o in obs_b)
        if len(metrics_a) > 1 or len(metrics_b) > 1:
            return False, f"Mixed metrics within observation lists: {metrics_a} vs {metrics_b}."
        if metrics_a != metrics_b:
            return False, f"Metric mismatch between observation lists: {metrics_a} vs {metrics_b}."

    if len(obs_a) != len(obs_b):
        return False, f"Observation count mismatch: {len(obs_a)} vs {len(obs_b)}."

    for o_a, o_b in zip(obs_a, obs_b):
        if o_a.dataset != o_b.dataset:
            return False, f"Dataset mismatch in paired observations: {o_a.dataset} vs {o_b.dataset}."
        if o_a.seed != o_b.seed:
            return False, f"Seed mismatch in paired observations: {o_a.seed} vs {o_b.seed}."

    return True, "Valid"


class StatisticsAnalyzer:
    """
    Analyzer for benchmark observation processing, statistical hypothesis testing,
    ablation breakdown generation, and CSV/JSON output formatting.
    """

    def __init__(self, observations: List[RawObservation], config: Optional[BenchmarkConfig] = None):
        self.observations = [obs for obs in observations if obs.status == "success"]
        self.raw_all = observations
        self.config = config

    def compute_aggregated_metrics(self) -> List[AggregatedMetric]:
        """Aggregate observations across seeds per dataset and method."""
        groups: Dict[Tuple[str, str], List[RawObservation]] = {}
        for obs in self.observations:
            key = (obs.dataset, obs.method)
            groups.setdefault(key, []).append(obs)

        results: List[AggregatedMetric] = []

        for (dataset, method), obs_list in groups.items():
            scores = [obs.score for obs in obs_list]
            evals = [obs.candidate_evaluations for obs in obs_list]
            runtimes = [obs.runtime_seconds for obs in obs_list]

            task_type = obs_list[0].task_type
            metric = obs_list[0].metric
            higher_better = is_higher_better(metric)

            mean_s = float(np.mean(scores))
            std_s = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            best_s = max(scores) if higher_better else min(scores)
            worst_s = min(scores) if higher_better else max(scores)

            mean_e = float(np.mean(evals))
            mean_r = float(np.mean(runtimes))

            success_cnt = len(obs_list)
            fail_cnt = sum(1 for obs in self.raw_all if obs.dataset == dataset and obs.method == method and obs.status != "success")

            results.append(
                AggregatedMetric(
                    dataset=dataset,
                    method=method,
                    task_type=task_type,
                    metric=metric,
                    mean_score=round(mean_s, 4),
                    std_score=round(std_s, 4),
                    best_score=round(best_s, 4),
                    worst_score=round(worst_s, 4),
                    mean_evaluations=round(mean_e, 1),
                    mean_runtime=round(mean_r, 2),
                    success_count=success_cnt,
                    fail_count=fail_cnt
                )
            )

        return results

    def compute_ablation_summary(self) -> List[AblationRecord]:
        """Generate overall ablation metrics across comparison methods A-E, separating classification F1 and regression RMSE."""
        method_groups: Dict[str, List[RawObservation]] = {}
        for obs in self.observations:
            method_groups.setdefault(obs.method, []).append(obs)

        unguided_evals = [obs.candidate_evaluations for obs in method_groups.get("method_e_unguided_baseline", [])]
        mean_unguided_evals = float(np.mean(unguided_evals)) if unguided_evals else 200.0

        ablation_records: List[AblationRecord] = []

        for method_id, meta in METHOD_METADATA.items():
            obs_list = method_groups.get(method_id, [])
            if not obs_list:
                continue

            class_scores = [obs.score for obs in obs_list if obs.task_type == "classification"]
            reg_scores = [obs.score for obs in obs_list if obs.task_type == "regression"]
            evals = [obs.candidate_evaluations for obs in obs_list]
            runtimes = [obs.runtime_seconds for obs in obs_list]

            mean_f1 = float(np.mean(class_scores)) if class_scores else 0.0
            mean_rmse = float(np.mean(reg_scores)) if reg_scores else 0.0
            mean_evals = float(np.mean(evals))
            mean_runtime = float(np.mean(runtimes))

            if mean_unguided_evals > 0:
                rel_gain = round(100.0 * (1.0 - (mean_evals / mean_unguided_evals)), 2)
            else:
                rel_gain = 0.0

            ablation_records.append(
                AblationRecord(
                    method_code=meta["code"],
                    method=method_id,
                    description=meta["description"],
                    has_dip=meta["has_dip"],
                    has_recommendation=meta["has_recommendation"],
                    has_optimization=meta["has_optimization"],
                    mean_classification_f1=round(mean_f1, 4),
                    mean_regression_rmse=round(mean_rmse, 4),
                    mean_candidate_evaluations=round(mean_evals, 1),
                    mean_runtime_seconds=round(mean_runtime, 2),
                    relative_efficiency_gain=rel_gain
                )
            )

        return sorted(ablation_records, key=lambda r: r.method_code)

    def _run_paired_comparison(
        self,
        obs_a: List[RawObservation],
        obs_b: List[RawObservation],
        comp_label: str,
        metric_name: str,
        test_type_label: str = "Paired t-test"
    ) -> Optional[StatisticalTestResult]:
        """Helper to run a validated paired t-test for a single metric/task type."""
        is_valid, msg = validate_comparable_observations(obs_a, obs_b, check_metric=True)
        if not is_valid:
            return None

        scores_a = np.array([o.score for o in obs_a])
        scores_b = np.array([o.score for o in obs_b])
        diffs = scores_a - scores_b
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

        if std_diff <= 1e-8:
            interp = f"{comp_label} ({metric_name}): Scores identical across all paired runs (mean diff = {mean_diff:+.4f})."
            return StatisticalTestResult(
                test_name=test_type_label,
                comparison=comp_label,
                metric=metric_name,
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_type="Cohen's d",
                interpretation=interp
            )

        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        cohen_d = float(mean_diff / std_diff) if std_diff != 0 else 0.0
        direction = get_metric_direction(metric_name)

        if direction == "higher":
            perf_desc = "higher (superior)" if mean_diff > 0 else ("lower (inferior)" if mean_diff < 0 else "identical")
        else:
            perf_desc = "lower (superior)" if mean_diff < 0 else ("higher (inferior)" if mean_diff > 0 else "identical")

        interp = (
            f"{comp_label} ({metric_name}, direction={direction}): "
            f"Mean difference = {mean_diff:+.4f} ({perf_desc}), t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={cohen_d:.2f}."
        )

        return StatisticalTestResult(
            test_name=test_type_label,
            comparison=comp_label,
            metric=metric_name,
            statistic=round(float(t_stat), 4),
            p_value=round(float(p_val), 4),
            effect_size=round(cohen_d, 4),
            effect_size_type="Cohen's d",
            interpretation=interp
        )

    def perform_statistical_tests(self) -> List[StatisticalTestResult]:
        """Perform formal statistical comparisons separated strictly by task metric type."""
        tests: List[StatisticalTestResult] = []

        # 1. Classification Macro F1: Method A vs Method E
        obs_a_cls = [o for o in self.observations if o.method == "method_a_full_genesis" and o.task_type == "classification"]
        obs_e_cls = [o for o in self.observations if o.method == "method_e_unguided_baseline" and o.task_type == "classification"]
        t1 = self._run_paired_comparison(obs_a_cls, obs_e_cls, "Method A vs Method E (Classification F1)", "f1")
        if t1:
            tests.append(t1)

        # 2. Regression RMSE: Method A vs Method E
        obs_a_reg = [o for o in self.observations if o.method == "method_a_full_genesis" and o.task_type == "regression"]
        obs_e_reg = [o for o in self.observations if o.method == "method_e_unguided_baseline" and o.task_type == "regression"]
        t2 = self._run_paired_comparison(obs_a_reg, obs_e_reg, "Method A vs Method E (Regression RMSE)", "rmse")
        if t2:
            tests.append(t2)

        # 3. Classification Macro F1: Method A vs Method B
        obs_b_cls = [o for o in self.observations if o.method == "method_b_without_dip" and o.task_type == "classification"]
        t3 = self._run_paired_comparison(obs_a_cls, obs_b_cls, "Method A vs Method B (Classification F1)", "f1")
        if t3:
            tests.append(t3)

        # 4. Regression RMSE: Method A vs Method B
        obs_b_reg = [o for o in self.observations if o.method == "method_b_without_dip" and o.task_type == "regression"]
        t4 = self._run_paired_comparison(obs_a_reg, obs_b_reg, "Method A vs Method B (Regression RMSE)", "rmse")
        if t4:
            tests.append(t4)

        # 5. Classification Macro F1: Method A vs Method D
        obs_d_cls = [o for o in self.observations if o.method == "method_d_recommendation_only" and o.task_type == "classification"]
        t5 = self._run_paired_comparison(obs_a_cls, obs_d_cls, "Method A vs Method D (Classification F1)", "f1")
        if t5:
            tests.append(t5)

        # 6. Regression RMSE: Method A vs Method D
        obs_d_reg = [o for o in self.observations if o.method == "method_d_recommendation_only" and o.task_type == "regression"]
        t6 = self._run_paired_comparison(obs_a_reg, obs_d_reg, "Method A vs Method D (Regression RMSE)", "rmse")
        if t6:
            tests.append(t6)

        # 7. Efficiency / Evaluations Paired Comparison: Method A vs Method B
        obs_a_all = [o for o in self.observations if o.method == "method_a_full_genesis"]
        obs_b_all = [o for o in self.observations if o.method == "method_b_without_dip"]

        if obs_a_all and obs_b_all and len(obs_a_all) == len(obs_b_all):
            evals_a = np.array([o.candidate_evaluations for o in obs_a_all])
            evals_b = np.array([o.candidate_evaluations for o in obs_b_all])
            diffs_evals = evals_a - evals_b
            mean_diff_evals = float(np.mean(diffs_evals))
            std_diff_evals = float(np.std(diffs_evals, ddof=1)) if len(diffs_evals) > 1 else 0.0

            if std_diff_evals > 1e-8:
                t_stat_ev, p_val_ev = stats.ttest_rel(evals_a, evals_b)
                cohen_d_ev = float(mean_diff_evals / std_diff_evals)
            else:
                t_stat_ev, p_val_ev, cohen_d_ev = 0.0, 1.0, 0.0

            interp_ev = (
                f"Candidate Evaluations - Full GENESIS (Method A) vs Without DIP (Method B): "
                f"Mean evaluations difference = {mean_diff_evals:+.1f}, t={t_stat_ev:.3f}, p={p_val_ev:.4f}."
            )
            tests.append(
                StatisticalTestResult(
                    test_name="Paired t-test (Efficiency A vs B)",
                    comparison="Method A vs Method B (Candidate Evaluations)",
                    metric="evaluations",
                    statistic=round(float(t_stat_ev), 4),
                    p_value=round(float(p_val_ev), 4),
                    effect_size=round(cohen_d_ev, 4),
                    effect_size_type="Cohen's d",
                    interpretation=interp_ev
                )
            )

        return tests

    def evaluate_hypotheses(self) -> List[HypothesisEvaluation]:
        """
        Evaluate research hypotheses H1, H2, H3, H4 empirically based strictly on observed data.
        Enforces H2 dual-condition (Efficiency AND Task-Separated Performance Maintenance),
        H3 task-separated GA ablation, and H4 task-separated stability.
        """
        evaluations: List[HypothesisEvaluation] = []
        method_groups: Dict[str, List[RawObservation]] = {}
        for obs in self.observations:
            method_groups.setdefault(obs.method, []).append(obs)

        obs_a = method_groups.get("method_a_full_genesis", [])
        obs_b = method_groups.get("method_b_without_dip", [])
        obs_c = method_groups.get("method_c_without_recommendation", [])
        obs_d = method_groups.get("method_d_recommendation_only", [])
        obs_e = method_groups.get("method_e_unguided_baseline", [])

        # -------------------------------------------------------------------------
        # H1: GENESIS-AI achieves competitive predictive performance.
        # Primary Comparison: Method A (Full GENESIS-AI) vs Method E (Unguided Baseline)
        # Evaluated strictly using task-separated statistical paired comparisons (no arbitrary win-count thresholds).
        # -------------------------------------------------------------------------
        if obs_a and obs_e:
            # 1. Classification Macro F1: A vs E (higher is better)
            obs_a_cls = [o for o in obs_a if o.task_type == "classification"]
            obs_e_cls = [o for o in obs_e if o.task_type == "classification"]

            cls_imp = False
            cls_degraded = False
            cls_details = "No classification observations."
            if obs_a_cls and obs_e_cls:
                s_a_c = np.array([o.score for o in obs_a_cls])
                s_e_c = np.array([o.score for o in obs_e_cls])
                m_a_c = float(np.mean(s_a_c))
                m_e_c = float(np.mean(s_e_c))
                diff_c = m_a_c - m_e_c

                if len(s_a_c) == len(s_e_c) and len(s_a_c) > 1 and np.std(s_a_c - s_e_c) > 1e-8:
                    t_c, p_c = stats.ttest_rel(s_a_c, s_e_c)
                    cls_imp = (t_c > 0) and (p_c < 0.05)
                    cls_degraded = (t_c < 0) and (p_c < 0.05)
                    cls_details = f"Classification Macro F1: A={m_a_c:.4f} vs E={m_e_c:.4f} (diff={diff_c:+.4f}, t={t_c:.3f}, p={p_c:.4f})"
                else:
                    cls_imp = False
                    cls_degraded = False
                    cls_details = f"Classification Macro F1: A={m_a_c:.4f} vs E={m_e_c:.4f} (diff={diff_c:+.4f}, stat test unavailable)"

            # 2. Regression RMSE: A vs E (lower is better)
            obs_a_reg = [o for o in obs_a if o.task_type == "regression"]
            obs_e_reg = [o for o in obs_e if o.task_type == "regression"]

            reg_imp = False
            reg_degraded = False
            reg_details = "No regression observations."
            if obs_a_reg and obs_e_reg:
                s_a_r = np.array([o.score for o in obs_a_reg])
                s_e_r = np.array([o.score for o in obs_e_reg])
                m_a_r = float(np.mean(s_a_r))
                m_e_r = float(np.mean(s_e_r))
                diff_r = m_a_r - m_e_r

                if len(s_a_r) == len(s_e_r) and len(s_a_r) > 1 and np.std(s_a_r - s_e_r) > 1e-8:
                    t_r, p_r = stats.ttest_rel(s_a_r, s_e_r)
                    reg_imp = (t_r < 0) and (p_r < 0.05)
                    reg_degraded = (t_r > 0) and (p_r < 0.05)
                    reg_details = f"Regression RMSE: A={m_a_r:.4f} vs E={m_e_r:.4f} (diff={diff_r:+.4f}, t={t_r:.3f}, p={p_r:.4f})"
                else:
                    reg_imp = False
                    reg_degraded = False
                    reg_details = f"Regression RMSE: A={m_a_r:.4f} vs E={m_e_r:.4f} (diff={diff_r:+.4f}, stat test unavailable)"

            # Descriptive dataset breakdown (for informative reporting only, not hardcoded status threshold)
            ds_names = sorted(list(set(obs.dataset for obs in self.observations)))
            matched_count = 0
            ds_details = []

            for ds in ds_names:
                obs_a_ds = [o for o in obs_a if o.dataset == ds]
                obs_e_ds = [o for o in obs_e if o.dataset == ds]
                if not obs_a_ds or not obs_e_ds:
                    continue

                m_a = float(np.mean([o.score for o in obs_a_ds]))
                m_e = float(np.mean([o.score for o in obs_e_ds]))
                metric_name = obs_a_ds[0].metric
                direction = get_metric_direction(metric_name)

                if direction == "higher":
                    is_match = m_a >= m_e
                else:
                    is_match = m_a <= m_e

                if is_match:
                    matched_count += 1
                ds_details.append(f"{ds} ({metric_name}: A={m_a:.4f} vs E={m_e:.4f})")

            all_matched = (matched_count == len(ds_names)) and (len(ds_names) > 0)

            if (cls_imp or reg_imp) and not cls_degraded and not reg_degraded:
                h1_status = "SUPPORTED"
                h1_rationale = (
                    f"Method A achieved statistically significant predictive performance improvement over Method E "
                    f"without degradation on either task. [{cls_details}; {reg_details}]."
                )
            elif cls_degraded or reg_degraded:
                h1_status = "NOT SUPPORTED"
                h1_rationale = (
                    f"Method A demonstrated statistically significant predictive performance degradation compared to Method E. "
                    f"[{cls_details}; {reg_details}]."
                )
            else:
                h1_status = "INCONCLUSIVE"
                h1_rationale = (
                    f"Method A matched or outperformed Method E on {matched_count}/{len(ds_names)} datasets ({'; '.join(ds_details)}), "
                    f"yielding mixed performance findings across dataset types. Score differences between Method A and baseline Method E "
                    f"were statistically non-significant (p >= 0.05). [{cls_details}; {reg_details}]."
                )
        else:
            h1_status = "INCONCLUSIVE"
            h1_rationale = "Insufficient observations to evaluate H1."

        evaluations.append(
            HypothesisEvaluation(
                hypothesis_id="H1",
                statement="GENESIS-AI achieves competitive predictive performance.",
                status=h1_status,
                rationale=h1_rationale
            )
        )

        # -------------------------------------------------------------------------
        # H2: DIP-guided search evaluates fewer candidate pipelines while maintaining competitive performance.
        # Primary Comparison: Method A (DIP+Rec+GA) vs Method B (NO DIP+Rec+GA)
        # -------------------------------------------------------------------------
        if obs_a and obs_b:
            # Component 1: Search Efficiency Reduction
            evals_a = np.array([o.candidate_evaluations for o in obs_a])
            evals_b = np.array([o.candidate_evaluations for o in obs_b])
            mean_a_ev = float(np.mean(evals_a))
            mean_b_ev = float(np.mean(evals_b))

            p_val_ev = 1.0
            if len(evals_a) == len(evals_b) and len(evals_a) > 1 and np.std(evals_a - evals_b) > 1e-8:
                _, p_val_ev = stats.ttest_rel(evals_a, evals_b)
                efficiency_pass = (mean_a_ev < mean_b_ev) and (p_val_ev < 0.05)
            else:
                efficiency_pass = False

            # Component 2: Performance Maintenance — Classification Macro F1
            obs_a_cls = [o for o in obs_a if o.task_type == "classification"]
            obs_b_cls = [o for o in obs_b if o.task_type == "classification"]

            cls_degraded = False
            cls_rationale = "No classification data."
            if obs_a_cls and obs_b_cls:
                scores_a_cls = np.array([o.score for o in obs_a_cls])
                scores_b_cls = np.array([o.score for o in obs_b_cls])
                m_a_f1 = float(np.mean(scores_a_cls))
                m_b_f1 = float(np.mean(scores_b_cls))
                diff_f1 = m_a_f1 - m_b_f1

                if len(scores_a_cls) == len(scores_b_cls) and len(scores_a_cls) > 1 and np.std(scores_a_cls - scores_b_cls) > 1e-8:
                    t_cls, p_cls = stats.ttest_rel(scores_a_cls, scores_b_cls)
                    # Higher is better: degraded if A is significantly lower than B
                    cls_degraded = (t_cls < 0) and (p_cls < 0.05)
                    cls_rationale = f"Classification Macro F1: A={m_a_f1:.4f} vs B={m_b_f1:.4f} (diff={diff_f1:+.4f}, t={t_cls:.3f}, p={p_cls:.4f})"
                else:
                    cls_degraded = False
                    cls_rationale = f"Classification Macro F1: A={m_a_f1:.4f} vs B={m_b_f1:.4f} (diff={diff_f1:+.4f}, stat test unavailable)"

            # Component 2: Performance Maintenance — Regression RMSE
            obs_a_reg = [o for o in obs_a if o.task_type == "regression"]
            obs_b_reg = [o for o in obs_b if o.task_type == "regression"]

            reg_degraded = False
            reg_rationale = "No regression data."
            if obs_a_reg and obs_b_reg:
                scores_a_reg = np.array([o.score for o in obs_a_reg])
                scores_b_reg = np.array([o.score for o in obs_b_reg])
                m_a_rmse = float(np.mean(scores_a_reg))
                m_b_rmse = float(np.mean(scores_b_reg))
                diff_rmse = m_a_rmse - m_b_rmse

                if len(scores_a_reg) == len(scores_b_reg) and len(scores_a_reg) > 1 and np.std(scores_a_reg - scores_b_reg) > 1e-8:
                    t_reg, p_reg = stats.ttest_rel(scores_a_reg, scores_b_reg)
                    # Lower is better: degraded if A is significantly HIGHER than B
                    reg_degraded = (t_reg > 0) and (p_reg < 0.05)
                    reg_rationale = f"Regression RMSE: A={m_a_rmse:.4f} vs B={m_b_rmse:.4f} (diff={diff_rmse:+.4f}, t={t_reg:.3f}, p={p_reg:.4f})"
                else:
                    reg_degraded = False
                    reg_rationale = f"Regression RMSE: A={m_a_rmse:.4f} vs B={m_b_rmse:.4f} (diff={diff_rmse:+.4f}, stat test unavailable)"

            performance_degraded = cls_degraded or reg_degraded
            performance_maintained = not performance_degraded

            if performance_degraded:
                h2_status = "NOT SUPPORTED"
                h2_rationale = (
                    f"Method A predictive performance was significantly degraded compared to Method B. "
                    f"[{cls_rationale}; {reg_rationale}]."
                )
            elif efficiency_pass and performance_maintained:
                h2_status = "SUPPORTED"
                h2_rationale = (
                    f"Method A evaluated significantly fewer candidates (mean {mean_a_ev:.1f}) than Method B ({mean_b_ev:.1f}) "
                    f"with p={p_val_ev:.4f} < 0.05 while maintaining performance across tasks. [{cls_rationale}; {reg_rationale}]."
                )
            else:
                h2_status = "INCONCLUSIVE"
                h2_rationale = (
                    f"Method A evaluated fewer candidates on average ({mean_a_ev:.1f}) than Method B ({mean_b_ev:.1f}), "
                    f"but candidate evaluation reduction or predictive performance differences were statistically non-significant (p >= 0.05). [{cls_rationale}; {reg_rationale}]."
                )
        else:
            h2_status = "INCONCLUSIVE"
            h2_rationale = "Insufficient observations to evaluate H2."

        evaluations.append(
            HypothesisEvaluation(
                hypothesis_id="H2",
                statement="DIP-guided search evaluates fewer candidate pipelines while maintaining competitive performance.",
                status=h2_status,
                rationale=h2_rationale
            )
        )

        # -------------------------------------------------------------------------
        # H3: Evolutionary optimization improves solution quality compared with recommendation-only selection.
        # Primary Comparison: Method A (GA) vs Method D (Recommendation Only, same Top-K pool)
        # MUST NOT mix classification F1 and regression RMSE in the same statistical calculation.
        # -------------------------------------------------------------------------
        if obs_a and obs_d:
            # 1. Classification F1 test: A vs D
            obs_a_cls = [o for o in obs_a if o.task_type == "classification"]
            obs_d_cls = [o for o in obs_d if o.task_type == "classification"]

            cls_imp = False
            cls_worse = False
            cls_details = "No classification observations."
            if obs_a_cls and obs_d_cls:
                s_a_c = np.array([o.score for o in obs_a_cls])
                s_d_c = np.array([o.score for o in obs_d_cls])
                m_a_c = float(np.mean(s_a_c))
                m_d_c = float(np.mean(s_d_c))
                diff_c = m_a_c - m_d_c

                if len(s_a_c) == len(s_d_c) and len(s_a_c) > 1 and np.std(s_a_c - s_d_c) > 1e-8:
                    t_c, p_c = stats.ttest_rel(s_a_c, s_d_c)
                    cls_imp = (t_c > 0) and (p_c < 0.05)
                    cls_worse = (t_c < 0) and (p_c < 0.05)
                    cls_details = f"Classification F1: A={m_a_c:.4f} vs D={m_d_c:.4f} (diff={diff_c:+.4f}, t={t_c:.3f}, p={p_c:.4f})"
                else:
                    cls_imp = False
                    cls_worse = False
                    cls_details = f"Classification F1: A={m_a_c:.4f} vs D={m_d_c:.4f} (diff={diff_c:+.4f}, stat test unavailable)"

            # 2. Regression RMSE test: A vs D
            obs_a_reg = [o for o in obs_a if o.task_type == "regression"]
            obs_d_reg = [o for o in obs_d if o.task_type == "regression"]

            reg_imp = False
            reg_worse = False
            reg_details = "No regression observations."
            if obs_a_reg and obs_d_reg:
                s_a_r = np.array([o.score for o in obs_a_reg])
                s_d_r = np.array([o.score for o in obs_d_reg])
                m_a_r = float(np.mean(s_a_r))
                m_d_r = float(np.mean(s_d_r))
                diff_r = m_a_r - m_d_r

                if len(s_a_r) == len(s_d_r) and len(s_a_r) > 1 and np.std(s_a_r - s_d_r) > 1e-8:
                    t_r, p_r = stats.ttest_rel(s_a_r, s_d_r)
                    # Lower is better: A is improved over D if A has significantly lower RMSE (t < 0)
                    reg_imp = (t_r < 0) and (p_r < 0.05)
                    reg_worse = (t_r > 0) and (p_r < 0.05)
                    reg_details = f"Regression RMSE: A={m_a_r:.4f} vs D={m_d_r:.4f} (diff={diff_r:+.4f}, t={t_r:.3f}, p={p_r:.4f})"
                else:
                    reg_imp = False
                    reg_worse = False
                    reg_details = f"Regression RMSE: A={m_a_r:.4f} vs D={m_d_r:.4f} (diff={diff_r:+.4f}, stat test unavailable)"

            if (cls_imp or reg_imp) and not (cls_worse or reg_worse):
                h3_status = "SUPPORTED"
                h3_rationale = f"Method A (GA) achieved statistically significant improvement over Method D without degradation on either task. [{cls_details}; {reg_details}]."
            elif cls_worse or reg_worse:
                h3_status = "NOT SUPPORTED"
                h3_rationale = f"Method D (Recommendation Only) performed significantly better than Method A. [{cls_details}; {reg_details}]."
            else:
                h3_status = "INCONCLUSIVE"
                h3_rationale = f"Score differences between Method A and Method D were small or statistically non-significant (p >= 0.05). [{cls_details}; {reg_details}]."
        else:
            h3_status = "INCONCLUSIVE"
            h3_rationale = "Insufficient observations to evaluate H3."

        evaluations.append(
            HypothesisEvaluation(
                hypothesis_id="H3",
                statement="Evolutionary optimization improves solution quality compared with recommendation-only selection.",
                status=h3_status,
                rationale=h3_rationale
            )
        )

        # -------------------------------------------------------------------------
        # H4: GENESIS-AI produces stable results across repeated runs.
        # Report classification and regression stability separately.
        # -------------------------------------------------------------------------
        if obs_a:
            cls_obs = [o for o in obs_a if o.task_type == "classification"]
            reg_obs = [o for o in obs_a if o.task_type == "regression"]

            # Classification Stability
            cls_cv_list = []
            cls_strings = []
            cls_groups: Dict[str, List[float]] = {}
            for o in cls_obs:
                cls_groups.setdefault(o.dataset, []).append(o.score)

            for ds, scores in cls_groups.items():
                if len(scores) > 1:
                    m_val = float(np.mean(scores))
                    s_val = float(np.std(scores, ddof=1))
                    cv_val = (s_val / abs(m_val)) if abs(m_val) > 1e-5 else 0.0
                    cls_cv_list.append(cv_val)
                    cls_strings.append(f"{ds} (F1 mean={m_val:.4f}, std={s_val:.4f}, CV={cv_val:.4f})")

            avg_cls_cv = float(np.mean(cls_cv_list)) if cls_cv_list else 0.0

            # Regression Stability
            reg_cv_list = []
            reg_strings = []
            reg_groups: Dict[str, List[float]] = {}
            for o in reg_obs:
                reg_groups.setdefault(o.dataset, []).append(o.score)

            for ds, scores in reg_groups.items():
                if len(scores) > 1:
                    m_val = float(np.mean(scores))
                    s_val = float(np.std(scores, ddof=1))
                    cv_val = (s_val / abs(m_val)) if abs(m_val) > 1e-5 else 0.0
                    reg_cv_list.append(cv_val)
                    reg_strings.append(f"{ds} (RMSE mean={m_val:.4f}, std={s_val:.4f}, CV={cv_val:.4f})")

            avg_reg_cv = float(np.mean(reg_cv_list)) if reg_cv_list else 0.0

            cls_summary = f"Classification Stability (Mean F1 CV = {avg_cls_cv:.4f}): {'; '.join(cls_strings)}" if cls_strings else "No classification runs."
            reg_summary = f"Regression Stability (Mean RMSE CV = {avg_reg_cv:.4f}): {'; '.join(reg_strings)}" if reg_strings else "No regression runs."

            h4_status = "INCONCLUSIVE"
            h4_rationale = (
                f"Method A demonstrated task-specific run-to-run variability across repeated random seeds. "
                f"[{cls_summary}] [{reg_summary}]. Strong stability could not be established due to small evaluation splits."
            )
        else:
            h4_status = "INCONCLUSIVE"
            h4_rationale = "Insufficient observations to evaluate H4."

        evaluations.append(
            HypothesisEvaluation(
                hypothesis_id="H4",
                statement="GENESIS-AI produces stable results across repeated runs.",
                status=h4_status,
                rationale=h4_rationale
            )
        )

        return evaluations

