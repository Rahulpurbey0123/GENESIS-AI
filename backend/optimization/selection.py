"""
Tournament Selection Operator for Evolutionary Pipeline Optimization Engine v1.0.

Selects parent chromosomes for breeding deterministically using tournament selection.
"""

from typing import List, Optional
import random
from backend.optimization.chromosome import Chromosome


def tournament_selection(
    population: List[Chromosome],
    fitnesses: List[float],
    tournament_size: int = 3,
    num_select: Optional[int] = None,
    rng: Optional[random.Random] = None
) -> List[Chromosome]:
    """
    Perform tournament selection to select parent chromosomes.

    Args:
        population: Current population of Chromosomes.
        fitnesses: List of corresponding fitness values.
        tournament_size: Number of individuals in each tournament (default 3).
        num_select: Total number of parent individuals to select (defaults to len(population)).
        rng: Optional random.Random instance.

    Returns:
        List of selected parent Chromosome instances.
    """
    if len(population) != len(fitnesses):
        raise ValueError("Population size and fitnesses length must match.")

    if not population:
        raise ValueError("Cannot perform selection on an empty population.")

    if tournament_size <= 0:
        tournament_size = 3

    t_size = min(tournament_size, len(population))
    n_select = num_select if num_select is not None else len(population)

    if rng is None:
        rng = random.Random()

    selected_parents: List[Chromosome] = []

    for _ in range(n_select):
        # Sample tournament_size indices
        indices = rng.sample(range(len(population)), k=t_size)
        # Find index with maximum fitness
        best_idx = max(indices, key=lambda idx: fitnesses[idx])
        selected_parents.append(population[best_idx].copy())

    return selected_parents
