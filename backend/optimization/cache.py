"""
Evaluation Cache for Evolutionary Pipeline Optimization Engine v1.0.

Prevents redundant model fitting and evaluation by caching chromosome fitness values.
"""

from typing import Dict, Any, Optional, Tuple
from backend.optimization.chromosome import Chromosome


class EvaluationCache:
    """
    Cache for storing and retrieving chromosome evaluation results across GA generations.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.evaluation_requests: int = 0
        self.unique_evaluations: int = 0
        self.cache_hits: int = 0

    def contains(self, chromosome: Chromosome) -> bool:
        """Check if chromosome result exists in cache."""
        return chromosome.get_canonical_key() in self._cache

    def get(self, chromosome: Chromosome) -> Optional[Dict[str, Any]]:
        """
        Lookup chromosome in cache.

        Updates evaluation_requests and cache_hits metrics.

        Returns:
            Dictionary containing cached result metrics or None if miss.
        """
        self.evaluation_requests += 1
        key = chromosome.get_canonical_key()
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        return None

    def put(self, chromosome: Chromosome, fitness: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Store fitness and metrics for a chromosome in cache.

        Updates unique_evaluations metric.
        """
        key = chromosome.get_canonical_key()
        if key not in self._cache:
            self.unique_evaluations += 1

        self._cache[key] = {
            "fitness": fitness,
            "metrics": metrics or {},
        }

    def clear(self) -> None:
        """Reset cache data and metrics counters."""
        self._cache.clear()
        self.evaluation_requests = 0
        self.unique_evaluations = 0
        self.cache_hits = 0

    def size(self) -> int:
        """Return number of unique configurations stored in cache."""
        return len(self._cache)
