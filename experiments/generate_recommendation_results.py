"""
Validation script to generate Recommendation Engine v1.1 results for all 5 software validation datasets.
"""

import json
from pathlib import Path
from backend.recommendation.engine import RecommendationEngine

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "test_datasets"
OUTPUT_FILE = BASE_DIR / "experiments" / "week3_recommendation_results.json"

TEST_DATASETS = [
    {"filename": "01_numerical_classification.csv", "target": "target"},
    {"filename": "02_categorical_heavy.csv", "target": "subscribed"},
    {"filename": "03_missing_values.csv", "target": "label"},
    {"filename": "04_imbalanced_classification.csv", "target": "is_fraud"},
    {"filename": "05_regression.csv", "target": "price"},
]


def run_recommendation_validation():
    engine = RecommendationEngine()
    results = {}

    print("=" * 90)
    print("GENESIS-AI WEEK 3 v1.1 RECOMMENDATION ENGINE SOFTWARE VALIDATION")
    print("=" * 90)

    for ds_info in TEST_DATASETS:
        filename = ds_info["filename"]
        target_col = ds_info["target"]
        csv_path = DATA_DIR / filename

        report = engine.recommend(csv_path, target_column=target_col, top_k=5)
        report_dict = report.model_dump()
        results[filename] = report_dict

        top_rec = report_dict["recommendations"][0] if report_dict["recommendations"] else {}
        print(f"\nDataset: {filename}")
        print(f"  Task Type:               {report_dict['task_type']}")
        print(f"  Complexity Score:        {report_dict['dataset_summary']['complexity_score']} ({report_dict['dataset_summary']['complexity_label']})")
        print(f"  Candidates Before:       {report_dict['candidate_count_before']}")
        print(f"  Candidates After Filter: {report_dict['candidate_count_after_filtering']}")
        print(f"  Filtering Reduction:     {report_dict['filtering_reduction'] * 100:.1f}%")
        print(f"  Configured Top-K:        {report_dict['top_k']}")
        print(f"  Recommended Count:       {report_dict['recommended_count']}")
        print(f"  Top-K Selection Ratio:   {report_dict['top_k_selection_ratio'] * 100:.1f}%")
        print(f"  Top Recommended:         {top_rec.get('pipeline_id')} ({top_rec.get('name')})")
        print(f"  Top Score:               {top_rec.get('score')}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print(f"Validation results saved to: {OUTPUT_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    run_recommendation_validation()
