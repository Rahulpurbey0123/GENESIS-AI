"""
Profiler module for GENESIS-AI Dataset Intelligence Profile (DIP) v1.1.

Extracts deterministic dataset characteristics: metadata, feature types (continuous numerical,
binary, categorical, boolean, datetime), quality metrics (feature & target missingness, duplicates),
outliers (IQR), skewness, correlation matrix, target/task properties, and dimensionality indicators.
"""

from typing import Dict, Any, List, Optional
import math
import numpy as np
import pandas as pd
from scipy import stats


# Threshold constant for task type detection heuristic
TASK_CLASSIFICATION_UNIQUE_THRESHOLD = 20
HIGH_CORRELATION_THRESHOLD = 0.90


def detect_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Detect data types for feature columns (excluding target).

    Categorizes features into:
    - continuous numerical (numeric dtype with > 2 unique values)
    - binary (any dtype with exactly 2 unique non-null values)
    - categorical (string/object with > 2 unique values)
    - boolean (native bool dtype)
    - datetime (datetime dtype or safely parseable datetime strings)
    """
    numeric_cols = []
    categorical_cols = []
    binary_cols = []
    boolean_cols = []
    datetime_cols = []

    for col in feature_cols:
        series = df[col].dropna()
        dtype = df[col].dtype
        n_unique = series.nunique()

        # Check Datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
            continue

        # Check Boolean
        if pd.api.types.is_bool_dtype(dtype):
            boolean_cols.append(col)
            binary_cols.append(col)
            continue

        # Check Binary (exactly 2 unique non-null values across any dtype)
        if n_unique == 2:
            binary_cols.append(col)
            continue

        # Check Numeric (continuous numerical with > 2 unique values)
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
            continue

        # Check Categorical / Object (> 2 unique values)
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            if len(series) > 0 and pd.api.types.is_string_dtype(dtype):
                sample = series.head(50)
                try:
                    parsed = pd.to_datetime(sample, format="ISO8601", errors="coerce")
                    if parsed.notna().sum() / len(sample) > 0.9:
                        datetime_cols.append(col)
                        continue
                except Exception:
                    pass
            categorical_cols.append(col)
            continue

        # Fallback to categorical
        categorical_cols.append(col)

    feature_count = len(feature_cols)
    numeric_count = len(numeric_cols)
    categorical_count = len(categorical_cols)
    binary_count = len(binary_cols)
    boolean_count = len(boolean_cols)
    datetime_count = len(datetime_cols)

    numeric_ratio = round(numeric_count / feature_count, 4) if feature_count > 0 else 0.0
    categorical_ratio = round(categorical_count / feature_count, 4) if feature_count > 0 else 0.0
    binary_ratio = round(binary_count / feature_count, 4) if feature_count > 0 else 0.0

    return {
        "numeric_features": numeric_count,
        "categorical_features": categorical_count,
        "binary_features": binary_count,
        "boolean_features": boolean_count,
        "datetime_features": datetime_count,
        "numeric_ratio": numeric_ratio,
        "categorical_ratio": categorical_ratio,
        "binary_ratio": binary_ratio,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "binary_columns": binary_cols,
        "boolean_columns": boolean_cols,
        "datetime_columns": datetime_cols,
    }


def analyze_missingness(df: pd.DataFrame, feature_cols: List[str], target_column: str) -> Dict[str, Any]:
    """
    Calculate missing value metrics separately for feature columns and target column.
    
    Feature missingness is computed exclusively over feature cells (N_rows * N_features).
    Target missingness is computed on the target column.
    """
    feature_cells = len(df) * len(feature_cols)
    total_feature_missing = int(df[feature_cols].isna().sum().sum()) if feature_cols else 0
    feature_missing_rate = round(total_feature_missing / feature_cells, 4) if feature_cells > 0 else 0.0

    per_feature_missing = {}
    cols_with_missing = 0
    max_col_missing_rate = 0.0

    for col in feature_cols:
        col_missing = int(df[col].isna().sum())
        col_missing_rate = round(col_missing / len(df), 4) if len(df) > 0 else 0.0
        per_feature_missing[col] = col_missing_rate
        if col_missing > 0:
            cols_with_missing += 1
            if col_missing_rate > max_col_missing_rate:
                max_col_missing_rate = col_missing_rate

    # Target missingness
    target_missing_count = int(df[target_column].isna().sum()) if target_column in df.columns else 0
    target_missing_rate = round(target_missing_count / len(df), 4) if len(df) > 0 else 0.0

    return {
        "total_missing": total_feature_missing,
        "missing_rate": feature_missing_rate,
        "columns_with_missing": cols_with_missing,
        "max_column_missing_rate": round(max_col_missing_rate, 4),
        "feature_missingness": {
            "total_missing": total_feature_missing,
            "missing_rate": feature_missing_rate,
            "columns_with_missing": cols_with_missing,
            "max_column_missing_rate": round(max_col_missing_rate, 4),
            "per_feature_missing_rates": per_feature_missing,
        },
        "target_missingness": {
            "missing_count": target_missing_count,
            "missing_rate": target_missing_rate,
        },
    }


def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate duplicate row count and rate."""
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = round(duplicate_rows / len(df), 4) if len(df) > 0 else 0.0
    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rate,
    }


def analyze_target(
    df: pd.DataFrame,
    target_column: str,
    unique_threshold: int = TASK_CLASSIFICATION_UNIQUE_THRESHOLD
) -> Dict[str, Any]:
    """
    Analyze target column and deterministically infer task type (classification vs regression).

    Heuristic Rule:
    1. Non-numeric dtypes (object, string, category, bool) -> classification.
    2. Continuous float numbers with fractional parts -> regression.
    3. Numeric dtypes where unique non-null count > unique_threshold (default 20) -> regression.
    4. Numeric dtypes where unique count ratio > 0.5 (with at least 10 unique values) -> regression.
    5. Otherwise -> classification.
    """
    target_series = df[target_column].dropna()
    unique_values = target_series.unique()
    n_unique = len(unique_values)
    dtype = df[target_column].dtype

    is_float = pd.api.types.is_float_dtype(dtype)
    is_numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)

    is_continuous_float = False
    if is_float and len(target_series) > 0:
        if not np.all(np.equal(np.mod(target_series, 1), 0)):
            is_continuous_float = True

    high_unique_ratio = (len(target_series) >= 10) and ((n_unique / len(target_series)) > 0.5)
    high_cardinality = (n_unique > unique_threshold) or high_unique_ratio

    if is_numeric and (is_continuous_float or high_cardinality):
        task_type = "regression"
    else:
        task_type = "classification"

    if task_type == "classification":
        value_counts = target_series.value_counts()
        class_dist = {str(k): int(v) for k, v in value_counts.items()}
        majority_class = str(value_counts.idxmax())
        minority_class = str(value_counts.idxmin())
        majority_count = value_counts.max()
        minority_count = value_counts.min()

        imbalance_ratio = round(float(majority_count / minority_count), 4) if minority_count > 0 else 1.0
        minority_pct = round(float((minority_count / len(target_series)) * 100), 2) if len(target_series) > 0 else 0.0

        # Class entropy calculation: H(Y) = - sum p_i log2(p_i)
        probs = value_counts / len(target_series)
        entropy_val = float(-np.sum(probs * np.log2(probs + 1e-12)))

        return {
            "name": target_column,
            "task_type": "classification",
            "class_count": n_unique,
            "class_distribution": class_dist,
            "majority_class": majority_class,
            "minority_class": minority_class,
            "imbalance_ratio": imbalance_ratio,
            "minority_percentage": minority_pct,
            "class_entropy": round(entropy_val, 4),
            "detection_heuristic": f"Target contains {n_unique} unique values -> classification.",
        }
    else:
        mean_val = float(target_series.mean())
        std_val = float(target_series.std()) if len(target_series) > 1 else 0.0
        median_val = float(target_series.median())
        min_val = float(target_series.min())
        max_val = float(target_series.max())
        skew_val = float(stats.skew(target_series, bias=False)) if len(target_series) > 2 else 0.0

        return {
            "name": target_column,
            "task_type": "regression",
            "class_count": None,
            "imbalance_ratio": None,
            "regression_stats": {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "median": round(median_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "skewness": round(skew_val, 4),
            },
            "detection_heuristic": f"Numeric target with high uniqueness ratio ({n_unique}/{len(target_series)}) -> regression.",
        }


def analyze_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """
    Perform IQR-based outlier detection on numerical features.

    IQR = Q3 - Q1
    Lower bound = Q1 - 1.5 * IQR
    Upper bound = Q3 + 1.5 * IQR
    """
    if not numeric_cols:
        return {
            "total_outliers": 0,
            "outlier_rate": 0.0,
            "columns_with_outliers": 0,
            "per_column_outlier_counts": {},
        }

    total_outliers = 0
    cols_with_outliers = 0
    per_col_outliers = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            per_col_outliers[col] = 0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 or pd.isna(iqr):
            per_col_outliers[col] = 0
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        per_col_outliers[col] = outlier_count
        total_outliers += outlier_count
        if outlier_count > 0:
            cols_with_outliers += 1

    total_numeric_cells = len(df) * len(numeric_cols)
    outlier_rate = round(total_outliers / total_numeric_cells, 4) if total_numeric_cells > 0 else 0.0

    return {
        "total_outliers": total_outliers,
        "outlier_rate": outlier_rate,
        "columns_with_outliers": cols_with_outliers,
        "per_column_outlier_counts": per_col_outliers,
    }


def analyze_skewness(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Calculate mean and max absolute skewness across numerical feature columns."""
    if not numeric_cols:
        return {
            "mean_absolute_skewness": 0.0,
            "max_absolute_skewness": 0.0,
            "per_feature_skewness": {},
        }

    per_feature_skew = {}
    abs_skews = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3 or series.nunique() <= 1:
            skew_val = 0.0
        else:
            try:
                skew_val = float(stats.skew(series, bias=False))
                if math.isnan(skew_val) or math.isinf(skew_val):
                    skew_val = 0.0
            except Exception:
                skew_val = 0.0
        
        per_feature_skew[col] = round(skew_val, 4)
        abs_skews.append(abs(skew_val))

    mean_abs_skew = round(float(np.mean(abs_skews)), 4) if abs_skews else 0.0
    max_abs_skew = round(float(np.max(abs_skews)), 4) if abs_skews else 0.0

    return {
        "mean_absolute_skewness": mean_abs_skew,
        "max_absolute_skewness": max_abs_skew,
        "per_feature_skewness": per_feature_skew,
    }


def analyze_correlation(
    df: pd.DataFrame,
    numeric_cols: List[str],
    threshold: float = HIGH_CORRELATION_THRESHOLD
) -> Dict[str, Any]:
    """
    Calculate Pearson correlation matrix metrics across numerical features using pairwise-complete correlation.

    Counts pairs with |r| >= threshold and records maximum absolute off-diagonal correlation.
    """
    if len(numeric_cols) < 2:
        return {
            "high_correlation_pairs": 0,
            "max_absolute_correlation": 0.0,
            "correlation_threshold": threshold,
        }

    numeric_df = df[numeric_cols]
    if len(numeric_df.dropna(how="all")) < 3:
        return {
            "high_correlation_pairs": 0,
            "max_absolute_correlation": 0.0,
            "correlation_threshold": threshold,
        }

    # Pairwise-complete correlation
    corr_matrix = numeric_df.corr(method="pearson").abs().values
    np.fill_diagonal(corr_matrix, 0.0)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Upper triangle indices
    triu_indices = np.triu_indices_from(corr_matrix, k=1)
    upper_corrs = corr_matrix[triu_indices]

    high_corr_count = int(np.sum(upper_corrs >= threshold))
    max_corr = float(np.max(upper_corrs)) if len(upper_corrs) > 0 else 0.0

    return {
        "high_correlation_pairs": high_corr_count,
        "max_absolute_correlation": round(max_corr, 4),
        "correlation_threshold": threshold,
    }


def compute_dataset_profile(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Orchestrate profile extraction for a validated DataFrame and target column."""
    feature_cols = [c for c in df.columns if c != target_column]

    types = detect_feature_types(df, feature_cols)
    missing = analyze_missingness(df, feature_cols, target_column)
    duplicates = analyze_duplicates(df)
    target = analyze_target(df, target_column)
    
    # For outliers, skewness, and correlation, analyze numeric features (excluding bool dtypes)
    numeric_for_stats = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(df[col].dtype) and not pd.api.types.is_bool_dtype(df[col].dtype)
    ]

    outliers = analyze_outliers_iqr(df, numeric_for_stats)
    skewness = analyze_skewness(df, numeric_for_stats)
    correlation = analyze_correlation(df, numeric_for_stats)

    feature_count = len(feature_cols)
    row_count = len(df)
    feature_to_sample_ratio = round(feature_count / row_count, 6) if row_count > 0 else 0.0

    return {
        "types": types,
        "missing": missing,
        "duplicates": duplicates,
        "target": target,
        "outliers": outliers,
        "skewness": skewness,
        "correlation": correlation,
        "dimensionality": {
            "feature_count": feature_count,
            "row_count": row_count,
            "feature_to_sample_ratio": feature_to_sample_ratio,
        },
    }
