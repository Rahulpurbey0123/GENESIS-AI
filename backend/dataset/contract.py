"""
Canonical Data Contract & Feature Isolation Module for GENESIS-AI v1.1.

Establishes authoritative invariants for ML pipelines:
1. target_column NOT IN feature_columns (P0 Target Leakage Prevention)
2. Identifier column detection and exclusion by default (P1 Identifier Audit)
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


IDENTIFIER_NAME_PATTERNS = [
    r"passengerid", r"customer_id", r"user_id", r"client_id", r"transaction_id",
    r"record_id", r"account_id", r"patient_id", r"subject_id", r"row_id",
    r"^id$", r"_id$", r"uuid", r"guid", r"index", r"seq_num"
]


def find_actual_column_name(df: pd.DataFrame, column_name: str) -> str:
    """
    Match column_name case-insensitively and stripped against df.columns.

    Args:
        df: Input pandas DataFrame.
        column_name: Candidate column name.

    Returns:
        Exact matching column name in df.columns.

    Raises:
        ValueError: If column_name is not found in df.columns.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("DataFrame is empty or invalid.")

    if not column_name or not isinstance(column_name, str):
        raise ValueError("Column name must be a non-empty string.")

    target_raw = column_name.strip()
    if target_raw in df.columns:
        return target_raw

    target_lower = target_raw.lower()
    for col in df.columns:
        if str(col).strip().lower() == target_lower:
            return col

    raise ValueError(f"Column '{column_name}' not found in DataFrame columns: {list(df.columns)}")


def detect_identifier_columns(df: pd.DataFrame, target_column: Optional[str] = None) -> List[str]:
    """
    Detect identifier-like columns in a DataFrame using auditable heuristics.

    Args:
        df: Input pandas DataFrame.
        target_column: Target column name to exclude from identifier detection.

    Returns:
        List of identified column names.
    """
    if df is None or df.empty:
        return []

    target_col = None
    if target_column:
        try:
            target_col = find_actual_column_name(df, target_column)
        except ValueError:
            target_col = target_column.strip()

    identifier_cols: List[str] = []
    n_rows = len(df)

    for col in df.columns:
        if target_col and col == target_col:
            continue

        col_str = str(col).strip().lower()

        # Check name pattern matches
        is_name_match = any(re.search(pat, col_str) for pat in IDENTIFIER_NAME_PATTERNS)

        # Check uniqueness ratio
        series = df[col].dropna()
        n_unique = series.nunique()
        uniqueness_ratio = (n_unique / n_rows) if n_rows > 0 else 0.0

        # Heuristic 1: Name match AND high cardinality or integer ID
        if is_name_match:
            identifier_cols.append(col)
            continue

        # Heuristic 2: Extremely high uniqueness ratio (>= 0.95) for integer/string columns with n_rows >= 20
        if n_rows >= 20 and uniqueness_ratio >= 0.95:
            if pd.api.types.is_integer_dtype(df[col].dtype) or pd.api.types.is_object_dtype(df[col].dtype) or pd.api.types.is_string_dtype(df[col].dtype):
                identifier_cols.append(col)

    return identifier_cols


def get_canonical_data_split(
    df: pd.DataFrame,
    target_column: str,
    exclude_identifiers: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], str, List[str]]:
    """
    Authoritative canonical data splitter for GENESIS-AI supervised learning pipelines.

    Guarantees:
    - Target column is matched case-insensitively and isolated as Series y.
    - Feature matrix X contains ONLY predictive features.
    - target_column NOT IN X.columns (Strict Invariant).
    - Optionally excludes identifier columns (PassengerId, customer_id, etc.) from X.

    Args:
        df: Input pandas DataFrame.
        target_column: Target column name.
        exclude_identifiers: Whether to exclude detected identifier columns from X (default True).

    Returns:
        Tuple of (X: DataFrame, y: Series, feature_names: List[str], actual_target_col: str, identifier_cols: List[str]).

    Raises:
        ValueError: If target column is invalid or zero predictive features remain.
    """
    actual_target = find_actual_column_name(df, target_column)
    identifier_cols = detect_identifier_columns(df, target_column=actual_target)

    # Determine predictive feature columns
    feature_cols = []
    for col in df.columns:
        if col == actual_target:
            continue
        if exclude_identifiers and col in identifier_cols:
            continue
        feature_cols.append(col)

    if not feature_cols:
        # Fallback if excluding identifiers stripped all features
        feature_cols = [c for c in df.columns if c != actual_target]

    X = df[feature_cols].copy()
    y = df[actual_target].copy()

    # Enforce strict invariant assertions
    assert actual_target not in X.columns, f"P0 TARGET LEAKAGE ERROR: Target '{actual_target}' found in feature matrix X!"
    assert actual_target not in feature_cols, f"P0 TARGET LEAKAGE ERROR: Target '{actual_target}' found in feature names list!"

    return X, y, feature_cols, actual_target, identifier_cols
