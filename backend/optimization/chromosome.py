"""
Chromosome Representation for Evolutionary Pipeline Optimization Engine v1.0.

Defines the individual structure comprising pipeline_id and model hyperparameters.
"""

from typing import Dict, Any, Optional
import json
import hashlib
from backend.optimization.schemas import ChromosomeDict
from backend.optimization.search_space import validate_hyperparameters, get_search_space


class Chromosome:
    """
    Individual chromosome in the Genetic Algorithm population.

    Represents a specific pipeline choice and its associated hyperparameter configuration.
    """

    def __init__(self, pipeline_id: str, hyperparameters: Dict[str, Any]):
        self.pipeline_id = pipeline_id
        self.hyperparameters = dict(hyperparameters)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chromosome to raw dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "hyperparameters": dict(self.hyperparameters),
        }

    def to_pydantic(self) -> ChromosomeDict:
        """Serialize chromosome to ChromosomeDict Pydantic model."""
        return ChromosomeDict(
            pipeline_id=self.pipeline_id,
            hyperparameters=dict(self.hyperparameters),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chromosome":
        """Reconstruct Chromosome instance from dictionary."""
        if "pipeline_id" not in data or "hyperparameters" not in data:
            raise ValueError("Invalid dictionary structure for Chromosome. Missing 'pipeline_id' or 'hyperparameters'.")
        return cls(pipeline_id=data["pipeline_id"], hyperparameters=data["hyperparameters"])

    def copy(self) -> "Chromosome":
        """Return a deep copy of the chromosome."""
        return Chromosome(pipeline_id=self.pipeline_id, hyperparameters=dict(self.hyperparameters))

    def get_canonical_key(self) -> str:
        """
        Generate deterministic canonical string representation for hashing and caching.
        """
        sorted_hp = json.dumps(self.hyperparameters, sort_keys=True)
        return f"{self.pipeline_id}::{sorted_hp}"

    def get_hash(self) -> str:
        """Generate SHA-256 hex digest of the canonical key."""
        key = self.get_canonical_key().encode("utf-8")
        return hashlib.sha256(key).hexdigest()

    def is_valid(self) -> bool:
        """Validate whether chromosome pipeline_id and hyperparameters are valid."""
        return validate_hyperparameters(self.pipeline_id, self.hyperparameters)

    def __repr__(self) -> str:
        return f"Chromosome(pipeline_id='{self.pipeline_id}', params={self.hyperparameters})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Chromosome):
            return False
        return self.get_canonical_key() == other.get_canonical_key()

    def __hash__(self) -> int:
        return hash(self.get_canonical_key())
