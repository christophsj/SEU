#!/usr/bin/env python3
"""
Run experiments for a single dataset (useful for testing)
Usage: python run_single_dataset.py <dataset_name> [num_runs]
Example: python run_single_dataset.py dbp_ja_en 1
"""

import sys
import os

# Import the main experiment functions
from run_experiments import (
    load_word_embeddings,
    run_single_experiment,
    save_results,
    DATASETS,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single_dataset.py <dataset_name> [num_runs]")
        print(f"Available datasets: {', '.join(DATASETS)}")
        sys.exit(1)

    dataset_name = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(DATASETS)}")
        sys.exit(1)

    print(f"Running {num_runs} experiment(s) for {dataset_name}...")

    # Load word embeddings
    word_vecs = load_word_embeddings()

    # Run experiments
    results = []
    for run_num in range(num_runs):
        result = run_single_experiment(dataset_name, "hybrid-level", word_vecs, run_num)
        results.append(result)

    # Save results
    save_results(results)

    print("\nExperiments completed!")


if __name__ == "__main__":
    main()
