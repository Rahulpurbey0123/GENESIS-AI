"""
GENESIS-AI Week 6 Controlled Experiment Runner.

Reads structured evidence from Week 5 (experiments/week5_explainability_results.json),
runs LLMService across all 5 software validation datasets for all 5 explanation modes
(simple, technical, prediction, research, pipeline), and exports structured output
to experiments/week6_llm_results.json.

Uses MockLLMClient by default for deterministic, reproducible, offline execution.
Opt-in to real OpenRouter API calls by setting LLM_PROVIDER=openrouter and OPENROUTER_API_KEY.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any

# Ensure project root is in import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.llm.config import LLMConfig
from backend.llm.service import LLMService
from backend.llm.modes import ExplanationMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("genesis.experiments.llm")


def run_llm_experiment() -> Dict[str, Any]:
    """Execute Week 6 LLM Explanation experiment across 5 datasets and 5 modes."""
    start_time = time.perf_counter()
    logger.info("=" * 100)
    logger.info("GENESIS-AI WEEK 6 LLM EXPLANATION LAYER EXPERIMENT")
    logger.info("=" * 100)

    exp_week5_path = os.path.join(PROJECT_ROOT, "experiments", "week5_explainability_results.json")
    if not os.path.exists(exp_week5_path):
        raise FileNotFoundError(f"Week 5 experiment results file missing: {exp_week5_path}")

    with open(exp_week5_path, "r", encoding="utf-8") as f:
        week5_data = json.load(f)

    # Initialize LLM Service (uses mock by default unless LLM_PROVIDER=openrouter)
    config = LLMConfig()
    service = LLMService(config=config)
    logger.info(f"Initialized LLMService with provider: '{config.provider}', model: '{config.model}'")

    results_by_dataset: Dict[str, Any] = {}
    modes = [m.value for m in ExplanationMode]

    for ds_name, ds_info in week5_data.items():
        logger.info(f"\nProcessing dataset: {ds_name}...")
        raw_evidence = ds_info.get("full_explanation", ds_info)

        mode_explanations: Dict[str, Any] = {}

        for mode_name in modes:
            logger.info(f"  Generating '{mode_name}' explanation...")
            llm_out = service.explain(raw_evidence, mode=mode_name)

            mode_explanations[mode_name] = {
                "explanation_mode": mode_name,
                "validation_status": llm_out.validation_status,
                "summary": llm_out.structured_explanation.summary,
                "model_explanation": llm_out.structured_explanation.model_explanation,
                "prediction_explanation": llm_out.structured_explanation.prediction_explanation,
                "important_features": llm_out.structured_explanation.important_features,
                "limitations": llm_out.structured_explanation.limitations,
                "evidence_used": llm_out.structured_explanation.evidence_used,
                "unsupported_claims": llm_out.structured_explanation.unsupported_claims,
                "warnings": llm_out.warnings,
                "runtime_seconds": llm_out.metadata.get("runtime_seconds", 0.0)
            }

        first_mode_out = mode_explanations["technical"]
        results_by_dataset[ds_name] = {
            "dataset": ds_info.get("dataset", ds_name),
            "task_type": ds_info.get("task_type", "classification"),
            "pipeline": ds_info.get("pipeline", "custom_pipeline"),
            "model": ds_info.get("model", "Unknown Estimator"),
            "metric": ds_info.get("metric", "score"),
            "score": ds_info.get("score", 0.0),
            "explanation_method": ds_info.get("method", "unsupported"),
            "llm_provider": config.provider,
            "llm_model": config.model,
            "modes_evaluated": modes,
            "explanations_by_mode": mode_explanations
        }

    total_time = round(time.perf_counter() - start_time, 2)
    logger.info("=" * 100)
    logger.info(f"Completed Week 6 Experiment across {len(results_by_dataset)} datasets in {total_time}s.")
    logger.info("=" * 100)

    # Save to experiments/week6_llm_results.json
    out_path = os.path.join(PROJECT_ROOT, "experiments", "week6_llm_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_by_dataset, f, indent=2)

    logger.info(f"Saved results to: {out_path}\n")
    return results_by_dataset


if __name__ == "__main__":
    run_llm_experiment()
