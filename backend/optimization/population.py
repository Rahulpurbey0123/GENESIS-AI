"""
Population Initialization for Evolutionary Pipeline Optimization Engine v1.0.

Generates initial population of valid Chromosomes for GENESIS mode and BASELINE mode.
"""

from typing import List, Optional
import random
from backend.recommendation.schemas import PipelineMetadata
from backend.optimization.chromosome import Chromosome
from backend.optimization.search_space import sample_random_hyperparameters


def generate_initial_population(
    candidate_pipelines: List[PipelineMetadata],
    pop_size: int = 20,
    rng: Optional[random.Random] = None
) -> List[Chromosome]:
    """
    Generate initial Genetic Algorithm population of valid Chromosomes.

    Args:
        candidate_pipelines: List of compatible candidate PipelineMetadata objects.
            - GENESIS Mode: Top-K recommended pipelines.
            - BASELINE Mode: All compatible pipelines after Stage 1 filtering.
        pop_size: Total population size to generate (default 20).
        rng: Optional random.Random instance for deterministic sampling.

    Returns:
        List of valid Chromosome instances.
    """
    if not candidate_pipelines:
        raise ValueError("Cannot initialize population with empty candidate pipelines list.")

    if pop_size <= 0:
        pop_size = 20

    if rng is None:
        rng = random.Random()

    population: List[Chromosome] = []

    # Ensure every candidate pipeline is represented at least once in initial population if pop_size >= len(candidates)
    shuffled_candidates = list(candidate_pipelines)
    rng.shuffle(shuffled_candidates)

    for i in range(pop_size):
        pipeline = shuffled_candidates[i % len(shuffled_candidates)]
        hp = sample_random_hyperparameters(pipeline.pipeline_id, rng=rng)
        chrom = Chromosome(pipeline_id=pipeline.pipeline_id, hyperparameters=hp)

        # Retry if somehow invalid
        attempts = 0
        while not chrom.is_valid() and attempts < 10:
            hp = sample_random_hyperparameters(pipeline.pipeline_id, rng=rng)
            chrom = Chromosome(pipeline_id=pipeline.pipeline_id, hyperparameters=hp)
            attempts += 1

        population.append(chrom)

    return population
