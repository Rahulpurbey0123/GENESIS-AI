"""
Dataset Intelligence Profile (DIP) builder and JSON serializer.

Orchestrates loading, validation, profiling, complexity calculation,
dataset SHA-256 hashing, and structured DIP JSON generation.
"""

import hashlib
import time
from typing import Dict, Any, Union, Optional
from pathlib import Path
import pandas as pd

from backend.dataset.loader import load_csv
from backend.dataset.validator import validate_dataset
from backend.dataset.profiler import compute_dataset_profile
from backend.dataset.complexity import compute_complexity_score, ComplexityWeights


DIP_VERSION = "1.1"


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute a deterministic, OS-independent canonical dataset-content SHA-256 hash.

    Procedure:
    1. Reindexes DataFrame columns in alphabetical sorted order.
    2. Excludes DataFrame index (index=False).
    3. Uses explicit Unix line endings (lineterminator='\\n') for cross-platform consistency.
    4. Serializes to UTF-8 encoded bytes.
    5. Computes SHA-256 hex digest.
    """
    sorted_df = df.reindex(sorted(df.columns), axis=1)
    csv_bytes = sorted_df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()



class DIPBuilder:
    """Builder class for Dataset Intelligence Profile (DIP) v1.1."""

    def __init__(
        self,
        dip_version: str = DIP_VERSION,
        complexity_weights: Optional[ComplexityWeights] = None
    ):
        self.dip_version = dip_version
        self.complexity_weights = complexity_weights

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_name: str = "dataset.csv"
    ) -> Dict[str, Any]:
        """
        Build a complete DIP from an already loaded pandas DataFrame and target column.

        Args:
            df: Input pandas DataFrame.
            target_column: Exact target column name.
            dataset_name: Display name or filename of the dataset.

        Returns:
            Structured DIP dictionary.
        """
        start_time = time.perf_counter()

        # Step 1: Validate dataset & target
        val_report = validate_dataset(df, target_column)
        target_col = val_report["target_column"]

        # Step 2: Extract Profile Metrics
        raw_profile = compute_dataset_profile(df, target_col)

        # Step 3: Compute Complexity Score
        complexity_info = compute_complexity_score(raw_profile, weights=self.complexity_weights)

        # Step 4: Compute Dataset SHA-256 Hash
        dataset_hash = compute_dataset_hash(df)

        # Step 5: Calculate Memory Usage
        memory_bytes = int(df.memory_usage(deep=True).sum())

        end_time = time.perf_counter()
        elapsed_ms = round((end_time - start_time) * 1000, 2)

        # Construct structured DIP v1.1 response
        dip_profile = {
            "dip_version": self.dip_version,
            "dataset_hash": dataset_hash,
            "dataset": {
                "name": dataset_name,
                "rows": len(df),
                "columns": len(df.columns),
                "feature_count": raw_profile["dimensionality"]["feature_count"],
                "memory_bytes": memory_bytes,
            },
            "schema": {
                "numeric_features": raw_profile["types"]["numeric_features"],
                "categorical_features": raw_profile["types"]["categorical_features"],
                "binary_features": raw_profile["types"]["binary_features"],
                "boolean_features": raw_profile["types"]["boolean_features"],
                "datetime_features": raw_profile["types"]["datetime_features"],
                "numeric_ratio": raw_profile["types"]["numeric_ratio"],
                "categorical_ratio": raw_profile["types"]["categorical_ratio"],
                "binary_ratio": raw_profile["types"]["binary_ratio"],
            },
            "quality": {
                "total_missing": raw_profile["missing"]["total_missing"],
                "missing_rate": raw_profile["missing"]["missing_rate"],
                "columns_with_missing": raw_profile["missing"]["columns_with_missing"],
                "max_column_missing_rate": raw_profile["missing"]["max_column_missing_rate"],
                "feature_missingness": raw_profile["missing"]["feature_missingness"],
                "target_missingness": raw_profile["missing"]["target_missingness"],
                "duplicate_rows": raw_profile["duplicates"]["duplicate_rows"],
                "duplicate_rate": raw_profile["duplicates"]["duplicate_rate"],
            },
            "statistics": {
                "total_outliers": raw_profile["outliers"]["total_outliers"],
                "outlier_rate": raw_profile["outliers"]["outlier_rate"],
                "columns_with_outliers": raw_profile["outliers"]["columns_with_outliers"],
                "mean_absolute_skewness": raw_profile["skewness"]["mean_absolute_skewness"],
                "max_absolute_skewness": raw_profile["skewness"]["max_absolute_skewness"],
                "high_correlation_pairs": raw_profile["correlation"]["high_correlation_pairs"],
                "max_absolute_correlation": raw_profile["correlation"]["max_absolute_correlation"],
            },
            "target": {
                "name": raw_profile["target"]["name"],
                "task_type": raw_profile["target"]["task_type"],
                "class_count": raw_profile["target"]["class_count"],
                "imbalance_ratio": raw_profile["target"]["imbalance_ratio"],
                "minority_percentage": raw_profile["target"].get("minority_percentage"),
                "class_entropy": raw_profile["target"].get("class_entropy"),
                "regression_stats": raw_profile["target"].get("regression_stats"),
            },
            "complexity_score": complexity_info["score"],
            "complexity_detail": {
                "label": complexity_info["complexity_label"],
                "normalized_components": complexity_info["normalized_components"],
                "weights": complexity_info["weights"],
            },
            "profiling_time_ms": elapsed_ms,
        }

        return dip_profile

    def build_from_file(
        self,
        file_source: Union[str, Path, bytes],
        target_column: str,
        filename: str = "dataset.csv"
    ) -> Dict[str, Any]:
        """
        Load CSV from file/bytes, validate, and build structured DIP.
        """
        df = load_csv(file_source, filename=filename)
        ds_name = filename
        if isinstance(file_source, (str, Path)):
            ds_name = Path(file_source).name
        return self.build_from_dataframe(df, target_column, dataset_name=ds_name)


def generate_dip(
    file_source_or_df: Union[str, Path, bytes, pd.DataFrame],
    target_column: str,
    dataset_name: str = "dataset.csv"
) -> Dict[str, Any]:
    """Convenience helper function to build DIP v1 profile."""
    builder = DIPBuilder()
    if isinstance(file_source_or_df, pd.DataFrame):
        return builder.build_from_dataframe(file_source_or_df, target_column, dataset_name=dataset_name)
    else:
        return builder.build_from_file(file_source_or_df, target_column, filename=dataset_name)
