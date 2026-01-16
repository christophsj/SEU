#!/usr/bin/env python3
"""
Run experiments for a single dataset (useful for testing)
Usage: python run_single_dataset.py <dataset_kg_folder> [num_runs]
Example: python run_single_dataset.py ja_en 1
Example: python run_single_dataset.py EN_FR_15K_V2 1
"""

import sys
import os

# Import the main experiment functions
from run_experiments import (
    load_word_embeddings,
    run_single_experiment,
    save_results,
    get_available_datasets,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single_dataset.py <dataset_kg_folder> [num_runs]")

        # Show available datasets
        available = get_available_datasets()
        if available:
            print("\nAvailable datasets:")
            for kg_path, trans_name in available:
                print(f"  - {kg_path} (using {trans_name} translations)")
        sys.exit(1)

    dataset_kg_path = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Find matching dataset
    available = get_available_datasets()
    dataset_match = None
    for kg_path, trans_name in available:
        if kg_path == dataset_kg_path:
            dataset_match = (kg_path, trans_name)
            break

    if not dataset_match:
        print(
            f"Error: Dataset '{dataset_kg_path}' not found or missing translated entity names"
        )
        print("\nAvailable datasets:")
        for kg_path, trans_name in available:
            print(f"  - {kg_path} (using {trans_name} translations)")
        sys.exit(1)

    kg_path, trans_name = dataset_match
    print(
        f"Running {num_runs} experiment(s) for {kg_path} (using {trans_name} translations)..."
    )

    # Load word embeddings
    word_vecs = load_word_embeddings()

    # Run experiments
    results = []
    for run_num in range(num_runs):
        result = run_single_experiment(
            kg_path, trans_name, "hybrid-level", word_vecs, run_num
        )
        results.append(result)

    # Save results
    save_results(results)

    print("\nExperiments completed!")


if __name__ == "__main__":
    main()
