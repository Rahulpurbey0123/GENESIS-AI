"""
Mutation Operators for Evolutionary Pipeline Optimization Engine v1.1.

Supports controlled model-family mutation (pipeline_id) and hyperparameter grid mutation.
"""

from typing import Optional, List
import random
from backend.optimization.chromosome import Chromosome
from backend.optimization.search_space import get_search_space, sample_random_hyperparameters


def mutate(
    chromosome: Chromosome,
    mutation_rate: float = 0.10,
    pipeline_mutation_rate: float = 0.10,
    allowed_pipeline_ids: Optional[List[str]] = None,
    rng: Optional[random.Random] = None
) -> Chromosome:
    """
    Mutate a chromosome via pipeline model-family mutation and hyperparameter mutation.

    Args:
        chromosome: Chromosome instance to mutate.
        mutation_rate: Probability of mutating each hyperparameter (default 0.10).
        pipeline_mutation_rate: Probability of mutating the pipeline_id model family (default 0.10).
        allowed_pipeline_ids: Allowed list of candidate pipeline_id strings for the current mode.
            - GENESIS Mode: Restricted Top-K candidate pool.
            - BASELINE Mode: Full compatible candidate pool.
        rng: Optional random.Random instance.

    Returns:
        Mutated valid Chromosome instance.
    """
    if rng is None:
        rng = random.Random()

    current_chrom = chromosome.copy()

    # Step 1: Model-Family Pipeline Mutation
    if rng.random() < pipeline_mutation_rate and allowed_pipeline_ids and len(allowed_pipeline_ids) > 1:
        candidates = [p for p in allowed_pipeline_ids if p != current_chrom.pipeline_id]
        if not candidates:
            candidates = allowed_pipeline_ids

        new_pipeline_id = rng.choice(candidates)

        # REGENERATE valid hyperparameters from scratch for the new model family
        new_hp = sample_random_hyperparameters(new_pipeline_id, rng=rng)
        candidate_chrom = Chromosome(pipeline_id=new_pipeline_id, hyperparameters=new_hp)

        if candidate_chrom.is_valid():
            current_chrom = candidate_chrom

    # Step 2: Hyperparameter Grid Mutation
    try:
        grid = get_search_space(current_chrom.pipeline_id)
    except ValueError:
        return current_chrom

    mutated_hp = dict(current_chrom.hyperparameters)
    hp_changed = False

    for param_name, param_values in grid.items():
        if rng.random() < mutation_rate:
            current_val = mutated_hp.get(param_name)
            opts = [v for v in param_values if v != current_val]
            if opts:
                mutated_hp[param_name] = rng.choice(opts)
            else:
                mutated_hp[param_name] = rng.choice(param_values)
            hp_changed = True

    if hp_changed:
        cand_chrom = Chromosome(
            pipeline_id=current_chrom.pipeline_id,
            hyperparameters=mutated_hp
        )
        if cand_chrom.is_valid():
            return cand_chrom

    return current_chrom
