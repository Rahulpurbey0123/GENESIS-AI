"""
Dataset & target validator module for GENESIS-AI Dataset Intelligence Profile (DIP).

Validates dataset dimensions, column existence, non-emptiness, and target column properties.
"""

from typing import Dict, Any, List
import pandas as pd


class DatasetValidationError(Exception):
    """Raised when dataset or target column fails validation rules."""
    pass


def validate_dataset(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Validate DataFrame structure and target column integrity.

    Args:
        df: Input pandas DataFrame.
        target_column: Exact name of the target column.

    Returns:
        Validation report dictionary with status and metadata.

    Raises:
        DatasetValidationError: If any validation check fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError("Dataset must be a pandas DataFrame.")

    if df.empty or len(df) == 0:
        raise DatasetValidationError("Dataset is empty.")

    if len(df.columns) < 2:
        raise DatasetValidationError(
            f"Dataset must have at least 2 columns (1 feature + 1 target). Found {len(df.columns)} column(s)."
        )

    if not target_column or not isinstance(target_column, str) or not target_column.strip():
        raise DatasetValidationError("Target column name must be a non-empty string.")

    target_col = target_column.strip()

    if target_col not in df.columns:
        raise DatasetValidationError(f"Target column '{target_col}' was not found.")

    target_series = df[target_col]
    non_null_target = target_series.dropna()

    if len(non_null_target) == 0:
        raise DatasetValidationError(f"Target column '{target_col}' contains only missing values.")

    unique_targets = non_null_target.unique()
    if len(unique_targets) < 2:
        raise DatasetValidationError(
            f"Target column '{target_col}' contains insufficient unique values (found {len(unique_targets)} unique value). A valid target requires at least 2 distinct values."
        )

    feature_cols: List[str] = [col for col in df.columns if col != target_col]

    return {
        "is_valid": True,
        "target_column": target_col,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
    }
