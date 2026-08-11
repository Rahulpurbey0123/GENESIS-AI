"""
Dataset Management & Data Splitting Module for GENESIS-AI Week 7 Research Evaluation.

Handles dataset path resolution, validation, task metadata extraction, and strict,
seed-governed Train / Validation / Test data partition generation.
"""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.dataset.loader import load_csv
from backend.dataset.validator import validate_dataset
from backend.dataset.dip import generate_dip
from backend.evaluation.schemas import DatasetSpec
from backend.evaluation.configuration import DATASETS_DIR, DEFAULT_DATASET_SPECS


class DatasetManagerError(Exception):
    """Exception raised for dataset loading or processing errors."""
    pass


class DatasetManager:
    """
    Manager for benchmark dataset ingestion, validation, DIP extraction, and data splitting.
    """

    def __init__(self, data_dir: Optional[Path] = None, specs: Optional[List[DatasetSpec]] = None):
        self.data_dir = data_dir or DATASETS_DIR
        self.specs = {spec.filename: spec for spec in (specs or DEFAULT_DATASET_SPECS)}

    def get_dataset_path(self, filename: str) -> Path:
        """Resolve full filesystem path for a dataset filename."""
        path = self.data_dir / filename
        if not path.exists():
            raise DatasetManagerError(f"Dataset file not found at path: {path}")
        return path

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load dataset as a pandas DataFrame."""
        path = self.get_dataset_path(filename)
        return load_csv(path, filename=filename)

    def get_spec(self, filename: str) -> DatasetSpec:
        """Retrieve dataset specification by filename."""
        if filename not in self.specs:
            raise DatasetManagerError(f"Dataset spec not registered for: {filename}")
        return self.specs[filename]

    def get_dip_profile(self, filename: str) -> Dict[str, Any]:
        """Extract Dataset Intelligence Profile (DIP v1.1) for dataset."""
        df = self.load_dataset(filename)
        spec = self.get_spec(filename)
        return generate_dip(df, target_column=spec.target, dataset_name=filename)

    def create_splits(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        seed: int,
        test_ratio: float = 0.20,
        val_ratio: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Partition dataset into strictly isolated Train (60%), Validation (20%), and Test (20%) splits.

        Args:
            df: Input DataFrame.
            target_column: Exact target column name.
            task_type: 'classification' or 'regression'.
            seed: Fixed random seed for split reproducibility.
            test_ratio: Isolated test split ratio (default 0.20).
            val_ratio: Validation split ratio (default 0.20).

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        val_report = validate_dataset(df, target_column)
        target_col = val_report["target_column"]

        X = df.drop(columns=[target_col])
        y = df[target_col]

        stratify_y = y if task_type.lower() == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2 else None

        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=test_ratio,
                random_state=seed,
                stratify=stratify_y
            )
        except Exception:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y,
                test_size=test_ratio,
                random_state=seed
            )

        val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
        stratify_tv = y_train_val if task_type.lower() == "classification" and y_train_val.nunique() > 1 and y_train_val.value_counts().min() >= 2 else None

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_ratio_adjusted,
                random_state=seed,
                stratify=stratify_tv
            )
        except Exception:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_ratio_adjusted,
                random_state=seed
            )

        return X_train, X_val, X_test, y_train, y_val, y_test
