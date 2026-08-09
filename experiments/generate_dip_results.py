"""Script to generate actual measured DIP v1.1 results across 5 software validation datasets."""

import json
from pathlib import Path
from backend.dataset.dip import generate_dip

DATA_DIR = Path("data/test_datasets")
OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_DATASETS_CONFIG = [
    ("01_numerical_classification.csv", "target"),
    ("02_categorical_heavy.csv", "subscribed"),
    ("03_missing_values.csv", "label"),
    ("04_imbalanced_classification.csv", "is_fraud"),
    ("05_regression.csv", "price"),
]


def main():
    results = {}
    print(f"{'Dataset':<35} | {'Task':<14} | {'Rows':<5} | {'Feats':<5} | {'Missing':<8} | {'Outliers':<8} | {'Imbalance':<9} | {'Complexity':<10} | {'Label':<8}")
    print("-" * 125)

    for filename, target_col in TEST_DATASETS_CONFIG:
        csv_path = DATA_DIR / filename
        dip = generate_dip(csv_path, target_column=target_col, dataset_name=filename)
        results[filename] = dip

        task_type = dip["target"]["task_type"]
        rows = dip["dataset"]["rows"]
        feats = dip["dataset"]["feature_count"]
        missing_rate = dip["quality"]["feature_missingness"]["missing_rate"]
        outlier_rate = dip["statistics"]["outlier_rate"]
        imbalance = dip["target"].get("imbalance_ratio")
        imbalance_str = f"{imbalance:.2f}" if imbalance is not None else "N/A"
        score = dip["complexity_score"]
        label = dip["complexity_detail"]["label"]

        print(f"{filename:<35} | {task_type:<14} | {rows:<5} | {feats:<5} | {missing_rate:<8.4f} | {outlier_rate:<8.4f} | {imbalance_str:<9} | {score:<10.2f} | {label:<8}")

    output_json_path = OUTPUT_DIR / "week2_dip_v1_1_results.json"
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved measured DIP v1.1 results to '{output_json_path}'")


if __name__ == "__main__":
    main()
