"""
Automated Research Analysis Pipeline for GENESIS-AI Week 4.2.1.

Parses multi-seed multi-Top-K raw results (experiments/week4_optimization_results.json),
computes corrected cache hit rates, separates classification (F1) and regression (RMSE) metrics,
calculates paired GENESIS vs BASELINE differences, evaluates baseline winner retention rates (baseline_best_in_genesis_top_k),
and exports:
- experiments/week4_analysis_summary.json
- experiments/week4_analysis_summary.csv
"""

import json
import csv
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "experiments" / "week4_optimization_results.json"
JSON_OUTPUT = BASE_DIR / "experiments" / "week4_analysis_summary.json"
CSV_OUTPUT = BASE_DIR / "experiments" / "week4_analysis_summary.csv"


def calculate_cache_metrics(unique_evaluations: int, cache_hits: int) -> tuple[int, float]:
    """
    Fix #1: Correct cache hit rate calculation formula.

    total_evaluation_requests = unique_evaluations + cache_hits
    cache_hit_rate = cache_hits / total_evaluation_requests (range: 0.0 to 1.0)
    """
    total_requests = unique_evaluations + cache_hits
    if total_requests == 0:
        return 0, 0.0
    hit_rate = round(cache_hits / total_requests, 4)
    return total_requests, hit_rate


def get_test_score(res_dict: dict, task_type: str) -> float:
    """Extract primary test metric score (F1 for classification, RMSE for regression)."""
    tp = res_dict.get("test_performance", {})
    if task_type == "classification":
        return float(tp.get("f1", 0.0))
    else:
        return float(tp.get("rmse", 0.0))


def analyze_results():
    if not INPUT_FILE.exists():
        print(f"Error: Input file '{INPUT_FILE}' not found. Run experiments/run_optimization.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    summary_data = {
        "per_dataset": {},
        "search_metrics_overall": {},
        "classification_metrics_overall": {},
        "regression_metrics_overall": {},
    }

    csv_rows = []
    # CSV Header
    csv_header = [
        "dataset", "seed", "mode", "top_k", "task_type",
        "candidate_count_before", "candidate_count_after", "candidate_space_reduction",
        "allowed_pipelines", "unique_evaluations", "cache_hits", "total_evaluation_requests",
        "cache_hit_rate", "best_pipeline", "best_val_fitness", "test_metric_name",
        "test_metric_value", "runtime_seconds", "baseline_best_pipeline",
        "baseline_best_in_genesis_top_k", "paired_test_metric_diff_vs_baseline",
        "paired_runtime_diff_vs_baseline", "paired_unique_eval_diff_vs_baseline"
    ]
    csv_rows.append(csv_header)

    print("=" * 110)
    print("GENESIS-AI WEEK 4.2.1 EXPERIMENTAL ANALYSIS & DIAGNOSTICS REPORT")
    print("=" * 110)

    # Accumulators for overall metrics by mode and task type
    search_metrics_acc = {}        # mode_key -> dict of lists for search cost metrics
    classification_acc = {}       # mode_key -> dict of lists for classification (F1)
    regression_acc = {}           # mode_key -> dict of lists for regression (RMSE)

    for dataset_name, seeds_dict in raw_data.items():
        print(f"\n" + "-" * 110)
        print(f"DATASET ANALYSIS: {dataset_name}")
        print("-" * 110)

        ds_summary = {}

        # Collect mode keys present for this dataset across seeds
        mode_keys = set()
        for seed_key, modes_data in seeds_dict.items():
            mode_keys.update(modes_data.keys())

        # Process each mode key (e.g. baseline, genesis_topk_2, genesis_topk_3, etc.)
        for mode_key in sorted(list(mode_keys)):
            runtimes = []
            unique_evals = []
            cache_hits_list = []
            total_reqs_list = []
            cache_hit_rates = []
            val_fitnesses = []
            test_scores = []
            reductions = []
            retention_flags = []
            paired_test_diffs = []
            paired_runtime_diffs = []
            paired_eval_diffs = []
            relative_rmse_diffs = []

            task_type = "classification"

            for seed_key, modes_data in seeds_dict.items():
                if mode_key not in modes_data or "baseline" not in modes_data:
                    continue

                m_res = modes_data[mode_key]
                b_res = modes_data["baseline"]

                task_type = m_res.get("task_type", "classification")
                m_score = get_test_score(m_res, task_type)
                b_score = get_test_score(b_res, task_type)

                m_runtime = float(m_res.get("runtime_seconds", 0.0))
                b_runtime = float(b_res.get("runtime_seconds", 0.0))

                m_unique = int(m_res.get("unique_evaluations", 0))
                b_unique = int(b_res.get("unique_evaluations", 0))

                chits = int(m_res.get("cache_hits", 0))
                total_reqs, ch_rate = calculate_cache_metrics(m_unique, chits)

                val_fit = float(m_res.get("best_fitness", float("-inf")))
                red = float(m_res.get("candidate_space_reduction", 0.0))
                retained = bool(m_res.get("baseline_best_in_genesis_top_k", True))

                # Fix #2: Paired metric differences calculated task-specifically
                if task_type == "classification":
                    test_diff = m_score - b_score  # Higher F1 is better; positive = GENESIS better
                else:  # regression
                    test_diff = m_score - b_score  # Lower RMSE is better; negative = GENESIS better
                    rel_rmse_diff = (test_diff / b_score) if b_score > 0 else 0.0
                    relative_rmse_diffs.append(rel_rmse_diff)

                runtime_diff = m_runtime - b_runtime
                eval_diff = m_unique - b_unique

                runtimes.append(m_runtime)
                unique_evals.append(m_unique)
                cache_hits_list.append(chits)
                total_reqs_list.append(total_reqs)
                cache_hit_rates.append(ch_rate)
                if val_fit != float("-inf"):
                    val_fitnesses.append(val_fit)
                test_scores.append(m_score)
                reductions.append(red)
                retention_flags.append(retained)

                paired_test_diffs.append(test_diff)
                paired_runtime_diffs.append(runtime_diff)
                paired_eval_diffs.append(eval_diff)

                # Record individual run row for CSV
                top_k_val = m_res.get("top_k", 10 if mode_key == "baseline" else 2)
                seed_num = int(seed_key.replace("seed_", ""))
                metric_name = "f1" if task_type == "classification" else "rmse"

                csv_rows.append([
                    dataset_name, seed_num, m_res.get("mode", "genesis"), top_k_val, task_type,
                    m_res.get("candidate_count_before", 0), m_res.get("candidate_count_after", 0),
                    round(red, 4), "|".join(m_res.get("candidate_pipeline_ids", [])),
                    m_unique, chits, total_reqs, ch_rate,
                    m_res.get("best_pipeline_id", ""), round(val_fit, 4) if val_fit != float("-inf") else "-inf",
                    metric_name, round(m_score, 4), round(m_runtime, 2),
                    m_res.get("baseline_best_pipeline", ""), retained,
                    round(test_diff, 4), round(runtime_diff, 2), eval_diff
                ])

            # Compute dataset-level mode statistics
            mean_ch_rate = round(float(np.mean(cache_hit_rates)), 4) if cache_hit_rates else 0.0

            mode_stats = {
                "task_type": task_type,
                "runs_count": len(runtimes),
                "candidate_space_reduction": round(float(np.mean(reductions)), 4) if reductions else 0.0,
                "winner_retention_rate": round(float(np.mean(retention_flags)), 4) if retention_flags else 0.0,
                "winner_retained_count": sum(retention_flags),
                "test_metric": {
                    "name": "f1" if task_type == "classification" else "rmse",
                    "mean": round(float(np.mean(test_scores)), 4) if test_scores else 0.0,
                    "std": round(float(np.std(test_scores)), 4) if test_scores else 0.0,
                    "median": round(float(np.median(test_scores)), 4) if test_scores else 0.0,
                    "min": round(float(np.min(test_scores)), 4) if test_scores else 0.0,
                    "max": round(float(np.max(test_scores)), 4) if test_scores else 0.0,
                },
                "runtime_seconds": {
                    "mean": round(float(np.mean(runtimes)), 2) if runtimes else 0.0,
                    "std": round(float(np.std(runtimes)), 2) if runtimes else 0.0,
                    "median": round(float(np.median(runtimes)), 2) if runtimes else 0.0,
                    "min": round(float(np.min(runtimes)), 2) if runtimes else 0.0,
                    "max": round(float(np.max(runtimes)), 2) if runtimes else 0.0,
                },
                "unique_evaluations": {
                    "mean": round(float(np.mean(unique_evals)), 2) if unique_evals else 0.0,
                    "std": round(float(np.std(unique_evals)), 2) if unique_evals else 0.0,
                    "median": round(float(np.median(unique_evals)), 2) if unique_evals else 0.0,
                    "min": int(np.min(unique_evals)) if unique_evals else 0,
                    "max": int(np.max(unique_evals)) if unique_evals else 0,
                },
                "cache_hits": {
                    "mean": round(float(np.mean(cache_hits_list)), 2) if cache_hits_list else 0.0,
                },
                "total_evaluation_requests": {
                    "mean": round(float(np.mean(total_reqs_list)), 2) if total_reqs_list else 0.0,
                },
                "cache_hit_rate": mean_ch_rate,
                "paired_diff_vs_baseline": {
                    "test_metric_diff_mean": round(float(np.mean(paired_test_diffs)), 4) if paired_test_diffs else 0.0,
                    "runtime_diff_mean": round(float(np.mean(paired_runtime_diffs)), 2) if paired_runtime_diffs else 0.0,
                    "unique_eval_diff_mean": round(float(np.mean(paired_eval_diffs)), 2) if paired_eval_diffs else 0.0,
                }
            }

            if task_type == "regression" and relative_rmse_diffs:
                mode_stats["paired_diff_vs_baseline"]["relative_rmse_diff_mean"] = round(float(np.mean(relative_rmse_diffs)), 4)

            ds_summary[mode_key] = mode_stats

            # Accumulate search cost metrics across all datasets
            if mode_key not in search_metrics_acc:
                search_metrics_acc[mode_key] = {
                    "runtimes": [], "unique_evals": [], "cache_hits": [],
                    "total_reqs": [], "cache_hit_rates": [], "reductions": [],
                    "retention_flags": [], "runtime_diffs": [], "eval_diffs": []
                }
            search_metrics_acc[mode_key]["runtimes"].extend(runtimes)
            search_metrics_acc[mode_key]["unique_evals"].extend(unique_evals)
            search_metrics_acc[mode_key]["cache_hits"].extend(cache_hits_list)
            search_metrics_acc[mode_key]["total_reqs"].extend(total_reqs_list)
            search_metrics_acc[mode_key]["cache_hit_rates"].extend(cache_hit_rates)
            search_metrics_acc[mode_key]["reductions"].extend(reductions)
            search_metrics_acc[mode_key]["retention_flags"].extend(retention_flags)
            search_metrics_acc[mode_key]["runtime_diffs"].extend(paired_runtime_diffs)
            search_metrics_acc[mode_key]["eval_diffs"].extend(paired_eval_diffs)

            # Accumulate predictive metrics strictly by task type
            if task_type == "classification":
                if mode_key not in classification_acc:
                    classification_acc[mode_key] = {"f1_scores": [], "f1_diffs": []}
                classification_acc[mode_key]["f1_scores"].extend(test_scores)
                classification_acc[mode_key]["f1_diffs"].extend(paired_test_diffs)
            else:  # regression
                if mode_key not in regression_acc:
                    regression_acc[mode_key] = {"rmse_scores": [], "rmse_diffs": [], "rel_rmse_diffs": []}
                regression_acc[mode_key]["rmse_scores"].extend(test_scores)
                regression_acc[mode_key]["rmse_diffs"].extend(paired_test_diffs)
                regression_acc[mode_key]["rel_rmse_diffs"].extend(relative_rmse_diffs)

            print(
                f"  [{mode_key:15s}] Red: {mode_stats['candidate_space_reduction']*100:5.1f}% | "
                f"Metric ({mode_stats['test_metric']['name']}): {mode_stats['test_metric']['mean']:8.4f} (std: {mode_stats['test_metric']['std']:5.4f}) | "
                f"Runtime: {mode_stats['runtime_seconds']['mean']:5.2f}s | "
                f"Unique Evals: {mode_stats['unique_evaluations']['mean']:5.1f} | "
                f"Cache Hit Rate: {mean_ch_rate*100:5.1f}% | "
                f"Winner Retained: {mode_stats['winner_retained_count']}/{mode_stats['runs_count']} ({mode_stats['winner_retention_rate']*100:5.1f}%)"
            )

        summary_data["per_dataset"][dataset_name] = ds_summary

    # Overall Search Cost Aggregation
    print("\n" + "=" * 110)
    print("OVERALL SEARCH COST & CANDIDATE REDUCTION METRICS ACROSS ALL DATASETS & SEEDS")
    print("=" * 110)

    for mode_key, acc in sorted(search_metrics_acc.items()):
        total_runs = len(acc["runtimes"])
        mean_red = round(float(np.mean(acc["reductions"])), 4) if acc["reductions"] else 0.0
        mean_ret = round(float(np.mean(acc["retention_flags"])), 4) if acc["retention_flags"] else 0.0
        mean_rt = round(float(np.mean(acc["runtimes"])), 2) if acc["runtimes"] else 0.0
        mean_rt_diff = round(float(np.mean(acc["runtime_diffs"])), 2) if acc["runtime_diffs"] else 0.0
        mean_ue = round(float(np.mean(acc["unique_evals"])), 2) if acc["unique_evals"] else 0.0
        mean_ue_diff = round(float(np.mean(acc["eval_diffs"])), 2) if acc["eval_diffs"] else 0.0
        mean_ch_rate = round(float(np.mean(acc["cache_hit_rates"])), 4) if acc["cache_hit_rates"] else 0.0

        summary_data["search_metrics_overall"][mode_key] = {
            "total_runs": total_runs,
            "candidate_space_reduction_mean": mean_red,
            "winner_retention_rate_mean": mean_ret,
            "runtime_seconds_mean": mean_rt,
            "paired_runtime_diff_mean": mean_rt_diff,
            "unique_evaluations_mean": mean_ue,
            "paired_unique_eval_diff_mean": mean_ue_diff,
            "cache_hit_rate_mean": mean_ch_rate,
        }

        print(
            f"  [{mode_key:15s}] Total Runs: {total_runs:3d} | "
            f"Mean Red: {mean_red*100:5.1f}% | "
            f"Mean Runtime: {mean_rt:5.2f}s (Diff: {mean_rt_diff:+5.2f}s) | "
            f"Mean Unique Evals: {mean_ue:5.1f} (Diff: {mean_ue_diff:+5.1f}) | "
            f"Mean Cache Hit Rate: {mean_ch_rate*100:5.1f}% | "
            f"Winner Retained: {mean_ret*100:5.1f}%"
        )

    # Classification Metrics Aggregation (Macro-F1)
    print("\n" + "-" * 110)
    print("CLASSIFICATION PREDICTIVE PERFORMANCE (MACRO-F1: HIGHER IS BETTER)")
    print("-" * 110)

    for mode_key, acc in sorted(classification_acc.items()):
        scores = acc["f1_scores"]
        diffs = acc["f1_diffs"]

        summary_data["classification_metrics_overall"][mode_key] = {
            "total_runs": len(scores),
            "mean_f1": round(float(np.mean(scores)), 4) if scores else 0.0,
            "std_f1": round(float(np.std(scores)), 4) if scores else 0.0,
            "median_f1": round(float(np.median(scores)), 4) if scores else 0.0,
            "mean_f1_diff_vs_baseline": round(float(np.mean(diffs)), 4) if diffs else 0.0,
            "median_f1_diff_vs_baseline": round(float(np.median(diffs)), 4) if diffs else 0.0,
        }

        c_stat = summary_data["classification_metrics_overall"][mode_key]
        print(
            f"  [{mode_key:15s}] Runs: {c_stat['total_runs']:2d} | "
            f"Mean F1: {c_stat['mean_f1']:6.4f} (std: {c_stat['std_f1']:5.4f}, median: {c_stat['median_f1']:6.4f}) | "
            f"Mean F1 Diff: {c_stat['mean_f1_diff_vs_baseline']:+6.4f}"
        )

    # Regression Metrics Aggregation (RMSE)
    print("\n" + "-" * 110)
    print("REGRESSION PREDICTIVE PERFORMANCE (RMSE: LOWER IS BETTER)")
    print("-" * 110)

    for mode_key, acc in sorted(regression_acc.items()):
        scores = acc["rmse_scores"]
        diffs = acc["rmse_diffs"]
        rel_diffs = acc["rel_rmse_diffs"]

        summary_data["regression_metrics_overall"][mode_key] = {
            "total_runs": len(scores),
            "mean_rmse": round(float(np.mean(scores)), 2) if scores else 0.0,
            "std_rmse": round(float(np.std(scores)), 2) if scores else 0.0,
            "median_rmse": round(float(np.median(scores)), 2) if scores else 0.0,
            "mean_rmse_diff_vs_baseline": round(float(np.mean(diffs)), 2) if diffs else 0.0,
            "median_rmse_diff_vs_baseline": round(float(np.median(diffs)), 2) if diffs else 0.0,
            "mean_relative_rmse_diff": round(float(np.mean(rel_diffs)), 4) if rel_diffs else 0.0,
        }

        r_stat = summary_data["regression_metrics_overall"][mode_key]
        print(
            f"  [{mode_key:15s}] Runs: {r_stat['total_runs']:2d} | "
            f"Mean RMSE: {r_stat['mean_rmse']:8.2f} (std: {r_stat['std_rmse']:7.2f}, median: {r_stat['median_rmse']:8.2f}) | "
            f"Mean RMSE Diff: {r_stat['mean_rmse_diff_vs_baseline']:+8.2f} (Relative: {r_stat['mean_relative_rmse_diff']*100:+5.2f}%)"
        )

    # Save JSON summary
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save CSV summary
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("\n" + "=" * 110)
    print(f"Summary JSON saved to: {JSON_OUTPUT}")
    print(f"Summary CSV saved to:  {CSV_OUTPUT}")
    print("=" * 100)


if __name__ == "__main__":
    analyze_results()
