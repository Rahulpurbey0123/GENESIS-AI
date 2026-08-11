"""
GENESIS-AI Week 7 Research Evaluation & Scientific Validation Package.

Provides datasets management, fairness protocols, metrics computation, benchmark runner,
baseline/ablation matrix execution, and statistical analysis tools.
"""

from backend.evaluation.schemas import (
    BenchmarkConfig,
    RawObservation,
    AggregatedMetric,
    AblationRecord,
    StatisticalTestResult,
    HypothesisEvaluation,
    BenchmarkRunSummary,
    DatasetSpec
)
from backend.evaluation.configuration import get_default_benchmark_config, METHOD_METADATA, DEFAULT_DATASET_SPECS
from backend.evaluation.datasets import DatasetManager
from backend.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_efficiency_metrics,
    get_primary_metric_name,
    is_higher_better,
    is_better_score
)
from backend.evaluation.protocols import FairnessProtocol
from backend.evaluation.baselines import BaselineExecutor
from backend.evaluation.runner import BenchmarkRunner
from backend.evaluation.statistics import StatisticsAnalyzer

__all__ = [
    "BenchmarkConfig",
    "RawObservation",
    "AggregatedMetric",
    "AblationRecord",
    "StatisticalTestResult",
    "HypothesisEvaluation",
    "BenchmarkRunSummary",
    "DatasetSpec",
    "get_default_benchmark_config",
    "METHOD_METADATA",
    "DEFAULT_DATASET_SPECS",
    "DatasetManager",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calculate_efficiency_metrics",
    "get_primary_metric_name",
    "is_higher_better",
    "is_better_score",
    "FairnessProtocol",
    "BaselineExecutor",
    "BenchmarkRunner",
    "StatisticsAnalyzer",
]
