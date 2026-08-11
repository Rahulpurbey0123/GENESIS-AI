"""
Fairness Protocol Enforcement Layer for GENESIS-AI Week 7 Research Evaluation.

Ensures strict experimental fairness across comparison methods:
- Identical dataset train/val/test splits.
- Identical candidate evaluation budgets.
- Fixed random seed propagation.
- Test set isolation invariants.
- Consistent metric definitions.
"""

from typing import Dict, Any, List, Optional, Tuple
from backend.evaluation.schemas import BenchmarkConfig, RawObservation


class ProtocolViolationError(Exception):
    """Raised when an experimental fairness protocol is violated."""
    pass


class FairnessProtocol:
    """
    Validator enforcing fairness constraints across benchmark comparison methods.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate central configuration for protocol compliance."""
        errors: List[str] = []

        if self.config.max_evaluations <= 0:
            errors.append(f"Invalid max_evaluations budget: {self.config.max_evaluations}. Must be positive.")

        if not self.config.seeds:
            errors.append("Random seed list cannot be empty.")

        if len(set(self.config.seeds)) != len(self.config.seeds):
            errors.append("Random seed list contains duplicates.")

        tot_ratio = round(self.config.train_ratio + self.config.val_ratio + self.config.test_ratio, 4)
        if abs(tot_ratio - 1.0) > 1e-5:
            errors.append(f"Split ratios must sum to 1.0. Got sum {tot_ratio}.")

        return len(errors) == 0, errors

    def validate_matched_runs(self, obs_a: RawObservation, obs_b: RawObservation) -> Tuple[bool, List[str]]:
        """Verify that two raw observations were evaluated under identical experimental conditions."""
        errors: List[str] = []

        if obs_a.dataset != obs_b.dataset:
            errors.append(f"Dataset mismatch: {obs_a.dataset} vs {obs_b.dataset}")

        if obs_a.seed != obs_b.seed:
            errors.append(f"Seed mismatch: {obs_a.seed} vs {obs_b.seed}")

        if obs_a.metric != obs_b.metric:
            errors.append(f"Metric mismatch: {obs_a.metric} vs {obs_b.metric}")

        if obs_a.task_type != obs_b.task_type:
            errors.append(f"Task type mismatch: {obs_a.task_type} vs {obs_b.task_type}")

        return len(errors) == 0, errors
