#!/bin/bash

# Master script to run all experiments
# This orchestrates the complete experimental pipeline

set -e  # Exit on error

echo "=================================================="
echo "SEU Entity Alignment - Experiment Automation"
echo "=================================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Virtual environment not active. Activating..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "ERROR: Virtual environment not found. Please run setup_environment.sh first."
        exit 1
    fi
fi

# Check if GloVe embeddings exist
if [ ! -f "glove.6B.300d.txt" ]; then
    echo "ERROR: GloVe embeddings not found. Please run setup_environment.sh first."
    exit 1
fi

# Create results directory
mkdir -p results

# Run experiments
echo -e "\n=================================================="
echo "Starting automated experiments..."
echo "This will run 3 experiments for each of 5 datasets"
echo "Total: 15 experiment runs"
echo "=================================================="

START_TIME=$(date +%s)

python run_experiments.py 2>&1 | tee results/experiment_log_$(date +%Y%m%d_%H%M%S).txt

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo -e "\n=================================================="
echo "All experiments completed!"
echo "Total time: ${DURATION_MIN} minutes"
echo "Results saved in the 'results' directory"
echo "=================================================="
