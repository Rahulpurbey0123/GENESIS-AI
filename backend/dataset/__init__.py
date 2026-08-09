"""Dataset intelligence package for GENESIS-AI."""

from backend.dataset.loader import load_csv, CSVLoaderError, EmptyFileError, MalformedCSVError
from backend.dataset.validator import validate_dataset, DatasetValidationError
from backend.dataset.profiler import compute_dataset_profile
from backend.dataset.complexity import compute_complexity_score
from backend.dataset.dip import DIPBuilder, generate_dip

__all__ = [
    "load_csv",
    "CSVLoaderError",
    "EmptyFileError",
    "MalformedCSVError",
    "validate_dataset",
    "DatasetValidationError",
    "compute_dataset_profile",
    "compute_complexity_score",
    "DIPBuilder",
    "generate_dip",
]

