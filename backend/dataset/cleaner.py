"""
Dataset Cleaning & Preprocessing module for GENESIS-AI ML Workflows.

Provides strict separation between target vector cleaning (row removal)
and feature matrix imputation (SimpleImputer) for supervised learning.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from backend.dataset.validator import DatasetValidationError


MISSING_STRING_TOKENS = {
    "", "nan", "NaN", "null", "None", "NA", "N/A", "n/a", "none", "NULL",
    "inf", "-inf", "Infinity", "-Infinity", "infinity", "-infinity", "?"
}


def clean_dataset_for_ml(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean raw dataset DataFrame for machine learning workflows.

    Rules:
    1. Target y: Rows with NaN, None, inf, -inf, or missing tokens in y are REMOVED.
       Target is NEVER imputed or fabricated.
    2. Features X: inf/-inf values in feature columns are converted to np.nan so that
       scikit-learn SimpleImputer can process them cleanly during pipeline fitting.
    3. Validates that sufficient valid rows (>0) and target classes (>=2) remain post-cleaning.

    Args:
        df: Input pandas DataFrame.
        target_column: Target column name.

    Returns:
        Cleaned pandas DataFrame containing valid rows and clean features/target.

    Raises:
        DatasetValidationError: If dataset is invalid, empty, or insufficient target rows remain.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise DatasetValidationError("Dataset is empty or not a valid DataFrame.")

    from backend.dataset.contract import find_actual_column_name
    try:
        target_col = find_actual_column_name(df, target_column)
    except ValueError as err:
        raise DatasetValidationError(str(err))

    df_clean = df.copy()


    # Step 1: Replace infinite numeric values with np.nan across entire DataFrame
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    # Step 2: Handle string/object missing tokens in target column
    target_s = df_clean[target_col]
    if pd.api.types.is_object_dtype(target_s) or pd.api.types.is_string_dtype(target_s):
        # Convert whitespace-only or recognized missing string tokens to np.nan
        str_mask = target_s.astype(str).str.strip().isin(MISSING_STRING_TOKENS)
        df_clean.loc[str_mask, target_col] = np.nan

    # Step 3: Remove rows where target y is NaN / missing / infinite
    df_clean = df_clean.dropna(subset=[target_col])

    # Step 4: Validate target column after row removal
    valid_y = df_clean[target_col]
    if len(valid_y) == 0:
        raise DatasetValidationError(
            f"Target column '{target_col}' contains only missing/invalid values. Zero valid target samples remain."
        )

    unique_y = valid_y.unique()
    if len(unique_y) < 2:
        raise DatasetValidationError(
            f"Target column '{target_col}' contains insufficient unique valid values (found {len(unique_y)} unique value). Machine learning requires at least 2 distinct target values."
        )

    # Step 5: Convert inf/-inf to np.nan in feature columns so Imputer handles them
    feature_cols = [col for col in df_clean.columns if col != target_col]
    if feature_cols:
        df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)

    return df_clean
