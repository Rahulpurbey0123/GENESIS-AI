"""
GENESIS-AI Week 7 Research Evaluation Benchmark Runner Script.

Executes the complete experimental matrix across 5 datasets, 5 comparison methods (A-E),
and 5 fixed random seeds (125 total execution runs).
Saves raw observations to experiments/week7_benchmark_results.json.
"""

import sys
import time
from pathlib import Path

# Ensure project root is in Python path
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.evaluation.configuration import get_default_benchmark_config
from backend.evaluation.runner import BenchmarkRunner


def main():
    print("=" * 100)
    print("GENESIS-AI WEEK 7 RESEARCH EVALUATION & SCIENTIFIC VALIDATION BENCHMARK")
    print("=" * 100)

    config = get_default_benchmark_config()

    print(f"Benchmark Version: {config.benchmark_version}")
    print(f"Datasets ({len(config.datasets)}): {[d.filename for d in config.datasets]}")
    print(f"Methods ({len(config.methods)}): {config.methods}")
    print(f"Seeds ({len(config.seeds)}): {config.seeds}")
    print(f"Candidate Budget per Run: {config.max_evaluations} evaluations")
    print(f"Raw Results Target: {config.output_benchmark_json}")
    print("=" * 100)

    runner = BenchmarkRunner(config=config)
    start_time = time.perf_counter()

    observations = runner.run_benchmark(verbose=True)

    elapsed = round(time.perf_counter() - start_time, 2)
    successful = sum(1 for obs in observations if obs.status == "success")
    failed = sum(1 for obs in observations if obs.status != "success")

    print("\n" + "=" * 100)
    print("BENCHMARK EXECUTION SUMMARY")
    print(f"Total Execution Time: {elapsed} seconds")
    print(f"Total Runs Attempted: {len(observations)}")
    print(f"Successful Runs:     {successful}")
    print(f"Failed Runs:         {failed}")
    print(f"Raw Results Saved To: {config.output_benchmark_json}")
    print("=" * 100)


if __name__ == "__main__":
    main()
