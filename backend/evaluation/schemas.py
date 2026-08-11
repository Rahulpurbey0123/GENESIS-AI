"""
Pydantic v2 Data Schemas for GENESIS-AI Week 7 Research Evaluation Layer.

Defines structured data contracts for experiment configuration, raw observations,
aggregated metrics, ablation records, statistical analysis results, and benchmark summaries.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class DatasetSpec(BaseModel):
    """Specification of a dataset included in the evaluation benchmark."""
    filename: str = Field(description="CSV filename of dataset.")
    target: str = Field(description="Target column name.")
    task_type: str = Field(description="Task type: 'classification' or 'regression'.")
    description: Optional[str] = Field(default=None, description="Short dataset description.")


class BenchmarkConfig(BaseModel):
    """Centralized configuration for Week 7 research evaluation experiments."""
    benchmark_version: str = Field(default="1.0", description="Benchmark version string.")
    datasets: List[DatasetSpec] = Field(default_factory=list, description="List of datasets to evaluate.")
    methods: List[str] = Field(
        default=["method_a_full_genesis", "method_b_without_dip", "method_c_without_recommendation", "method_d_recommendation_only", "method_e_unguided_baseline"],
        description="List of comparison method identifiers."
    )
    seeds: List[int] = Field(default=[42, 123, 456, 789, 2024], description="Fixed random seeds for repeated runs.")
    max_evaluations: int = Field(default=200, description="Candidate evaluation budget per run.")
    top_k: int = Field(default=2, description="Top-K selection parameter for GENESIS recommendation.")
    population_size: int = Field(default=20, description="GA population size.")
    generations: int = Field(default=10, description="GA generation count.")
    crossover_rate: float = Field(default=0.80, description="GA crossover rate.")
    mutation_rate: float = Field(default=0.10, description="GA hyperparameter mutation rate.")
    pipeline_mutation_rate: float = Field(default=0.10, description="GA model family mutation rate.")
    train_ratio: float = Field(default=0.60, description="Train split ratio.")
    val_ratio: float = Field(default=0.20, description="Validation split ratio.")
    test_ratio: float = Field(default=0.20, description="Test split ratio.")
    output_benchmark_json: str = Field(default="experiments/week7_benchmark_results.json", description="Raw results output path.")
    output_summary_csv: str = Field(default="experiments/week7_summary.csv", description="Summary CSV output path.")
    output_ablation_json: str = Field(default="experiments/week7_ablation_results.json", description="Ablation JSON output path.")


class RawObservation(BaseModel):
    """Raw observation record from a single experimental execution run."""
    dataset: str = Field(description="Dataset filename.")
    task_type: str = Field(description="Task type: 'classification' or 'regression'.")
    method: str = Field(description="Comparison method identifier (e.g. method_a_full_genesis).")
    seed: int = Field(description="Random seed used for the run.")
    metric: str = Field(description="Evaluation metric name (e.g. 'f1' or 'rmse').")
    score: float = Field(description="Measured test score on isolated test split.")
    candidate_evaluations: int = Field(description="Number of candidate pipeline evaluations performed.")
    unique_evaluations: int = Field(default=0, description="Number of unique pipeline candidate evaluations.")
    runtime_seconds: float = Field(description="Measured execution runtime in seconds.")
    best_configuration: Dict[str, Any] = Field(default_factory=dict, description="Best pipeline and hyperparameters found.")
    status: str = Field(default="success", description="Execution status: 'success' or 'failed'.")
    error: Optional[str] = Field(default=None, description="Error message if run failed.")


class AggregatedMetric(BaseModel):
    """Aggregated statistics across repeated seed runs for a dataset/method combination."""
    dataset: str = Field(description="Dataset filename.")
    method: str = Field(description="Comparison method identifier.")
    task_type: str = Field(description="Task type.")
    metric: str = Field(description="Primary metric name.")
    mean_score: float = Field(description="Mean test score across valid seeds.")
    std_score: float = Field(description="Standard deviation of test scores.")
    best_score: float = Field(description="Best (highest for F1, lowest for RMSE) score observed.")
    worst_score: float = Field(description="Worst score observed.")
    mean_evaluations: float = Field(description="Mean candidate pipeline evaluations.")
    mean_runtime: float = Field(description="Mean execution runtime in seconds.")
    success_count: int = Field(description="Number of successful seed runs.")
    fail_count: int = Field(description="Number of failed seed runs.")


class AblationRecord(BaseModel):
    """Record summarizing an ablation configuration variant for comparison."""
    method_code: str = Field(description="Method letter code (A, B, C, D, E).")
    method: str = Field(description="Method identifier.")
    description: str = Field(description="Human-readable component breakdown.")
    has_dip: bool = Field(description="Whether DIP profile is used.")
    has_recommendation: bool = Field(description="Whether Top-K Recommendation Engine space pruning is used.")
    has_optimization: bool = Field(description="Whether Evolutionary Optimization GA search is used.")
    mean_classification_f1: float = Field(description="Average Macro F1 score across classification datasets.")
    mean_regression_rmse: float = Field(description="Average RMSE across regression datasets.")
    mean_candidate_evaluations: float = Field(description="Average candidate evaluations across all datasets.")
    mean_runtime_seconds: float = Field(description="Average execution runtime in seconds across all datasets.")
    relative_efficiency_gain: float = Field(description="Percentage reduction in evaluations compared to full unguided baseline.")


class StatisticalTestResult(BaseModel):
    """Statistical test or comparison result between two methods."""
    test_name: str = Field(description="Name of statistical method used (e.g. 'Paired t-test', 'Wilcoxon signed-rank', 'Descriptive').")
    comparison: str = Field(description="Pairwise comparison label (e.g. 'Method A vs Method E').")
    metric: str = Field(description="Target metric evaluated.")
    statistic: Optional[float] = Field(default=None, description="Test statistic value if computed.")
    p_value: Optional[float] = Field(default=None, description="p-value if formal hypothesis testing performed.")
    effect_size: Optional[float] = Field(default=None, description="Calculated effect size (e.g. Cohen's d).")
    effect_size_type: Optional[str] = Field(default=None, description="Effect size type string.")
    interpretation: str = Field(description="Scientific interpretation of the statistical result.")


class HypothesisEvaluation(BaseModel):
    """Evaluation record for a specific research hypothesis."""
    hypothesis_id: str = Field(description="Hypothesis ID (H1, H2, H3, H4).")
    statement: str = Field(description="Formal hypothesis statement.")
    status: str = Field(description="Outcome: 'SUPPORTED', 'NOT SUPPORTED', or 'INCONCLUSIVE'.")
    rationale: str = Field(description="Empirical evidence and technical reasoning supporting the status.")


class BenchmarkRunSummary(BaseModel):
    """Complete summary container for the Week 7 research evaluation benchmark."""
    version: str = Field(default="1.0", description="Benchmark summary version.")
    git_commit: str = Field(default="unknown", description="Git commit hash.")
    python_version: str = Field(default="unknown", description="Python version environment.")
    config: BenchmarkConfig = Field(description="Benchmark configuration details.")
    total_observations: int = Field(description="Total raw execution runs attempted.")
    successful_observations: int = Field(description="Total successful execution runs.")
    failed_observations: int = Field(description="Total failed execution runs.")
    aggregated_metrics: List[AggregatedMetric] = Field(default_factory=list, description="Dataset/method aggregated metrics.")
    ablation_summary: List[AblationRecord] = Field(default_factory=list, description="Summary across ablation matrix A-E.")
    statistical_tests: List[StatisticalTestResult] = Field(default_factory=list, description="Statistical comparison findings.")
    hypothesis_evaluations: List[HypothesisEvaluation] = Field(default_factory=list, description="Status for H1, H2, H3, H4.")
