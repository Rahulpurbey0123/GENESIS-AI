"""
Crossover Operator for Evolutionary Pipeline Optimization Engine v1.0.

Recombines parent chromosomes to produce valid offspring chromosomes.
"""

from typing import Tuple, Optional
import random
from backend.optimization.chromosome import Chromosome
from backend.optimization.search_space import get_search_space, sample_random_hyperparameters


def crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    crossover_rate: float = 0.80,
    rng: Optional[random.Random] = None
) -> Tuple[Chromosome, Chromosome]:
    """
    Perform structural and hyperparameter crossover between two parent chromosomes.

    Args:
        parent1: First parent Chromosome.
        parent2: Second parent Chromosome.
        crossover_rate: Probability of performing crossover (default 0.80).
        rng: Optional random.Random instance.

    Returns:
        Tuple of two offspring Chromosome instances.
    """
    if rng is None:
        rng = random.Random()

    off1 = parent1.copy()
    off2 = parent2.copy()

    if rng.random() > crossover_rate:
        return off1, off2

    # Case 1: Same model pipeline structure -> swap hyperparameter values
    if parent1.pipeline_id == parent2.pipeline_id:
        hp1 = dict(parent1.hyperparameters)
        hp2 = dict(parent2.hyperparameters)

        keys = list(hp1.keys())
        for key in keys:
            if key in hp2 and rng.random() < 0.5:
                hp1[key], hp2[key] = hp2[key], hp1[key]

        cand1 = Chromosome(pipeline_id=parent1.pipeline_id, hyperparameters=hp1)
        cand2 = Chromosome(pipeline_id=parent2.pipeline_id, hyperparameters=hp2)

        if cand1.is_valid():
            off1 = cand1
        if cand2.is_valid():
            off2 = cand2

    # Case 2: Different pipeline structures -> swap structural pipeline_id
    else:
        cand1 = Chromosome(pipeline_id=parent2.pipeline_id, hyperparameters=dict(parent2.hyperparameters))
        cand2 = Chromosome(pipeline_id=parent1.pipeline_id, hyperparameters=dict(parent1.hyperparameters))

        if cand1.is_valid():
            off1 = cand1
        if cand2.is_valid():
            off2 = cand2

    return off1, off2
