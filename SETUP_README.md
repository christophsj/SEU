# SEU Entity Alignment - Automated Experiment Setup

This repository contains the code for "From Alignment to Assignment: Frustratingly Simple but Effective Unsupervised Entity Alignment without Neural Networks" with automated scripts for running experiments on your datasets.

## Quick Start

### 1. Environment Setup

Run the setup script to create a virtual environment, install dependencies, and download GloVe embeddings:

```bash
./setup_environment.sh
```

This will:
- Create a Python virtual environment (`venv/`)
- Install all required packages from `requirements.txt`
- Download GloVe 300d word embeddings (~822MB)

### 2. Run Experiments

After setup is complete, run all experiments with:

```bash
./run_all_experiments.sh
```

This will:
- Run experiments 3 times for each of the 5 datasets
- Test both Hungarian and Sinkhorn methods
- Save results to the `results/` directory
- Generate summary statistics

**Note:** Running all experiments may take 30-60 minutes depending on your machine.

## Datasets

The repository includes 5 entity alignment datasets:

1. `dbp_fr_en` - DBpedia French-English
2. `dbp_ja_en` - DBpedia Japanese-English
3. `dbp_zh_en` - DBpedia Chinese-English
4. `srprs_de_en` - SRPRS German-English
5. `srprs_fr_en` - SRPRS French-English

Each dataset contains:
- `ent_ids_1` - Entity IDs in source KG
- `ent_ids_2` - Entity IDs in target KG
- `ref_ent_ids` - Reference entity alignments
- `triples_1` - Relation triples in source KG
- `triples_2` - Relation triples in target KG

## Results

Results are saved in the `results/` directory:

- `all_experiments_YYYYMMDD_HHMMSS.json` - Complete results for all runs
- `summary_YYYYMMDD_HHMMSS.json` - Summary statistics (mean ± std)
- `experiment_log_YYYYMMDD_HHMMSS.txt` - Complete execution log

### Metrics Reported

For each dataset and run:
- **Hungarian Algorithm**: Hits@1
- **Sinkhorn Operation**: Hits@1, Hits@10, MRR

Summary statistics include mean, standard deviation, min, and max across the 3 runs.

## Manual Execution

If you prefer to run experiments manually:

```bash
# Activate environment
source venv/bin/activate

# Run Python script directly
python run_experiments.py

# Or use Jupyter notebook
jupyter notebook main.ipynb
```

## Configuration

You can modify experiment parameters in `run_experiments.py`:

```python
SEED = 12345              # Random seed
NUM_RUNS = 3              # Number of runs per dataset
GRAPH_DEPTH = 2           # Graph propagation depth
FEATURE_MODES = ["hybrid-level"]  # Feature type
```

## Requirements

- Python 3.6.5+
- TensorFlow 2.4.1
- NumPy, SciPy, Numba, tqdm
- ~1GB disk space for GloVe embeddings
- ~2-4GB RAM depending on dataset

## File Structure

```
SEU/
├── setup_environment.sh      # Environment setup script
├── run_all_experiments.sh    # Master automation script
├── run_experiments.py        # Python experiment runner
├── requirements.txt          # Python dependencies
├── main.ipynb               # Original Jupyter notebook
├── utils.py                 # Utility functions
├── KGs/                     # Knowledge graph datasets
├── translated_ent_name/     # Entity name translations
├── results/                 # Experiment results (generated)
└── venv/                    # Virtual environment (generated)
```

## Troubleshooting

### GloVe Download Issues

If automatic download fails, manually download from:
http://nlp.stanford.edu/data/glove.6B.zip

Extract `glove.6B.300d.txt` to the repository root.

### Memory Issues

If you run out of memory:
- Run experiments for individual datasets
- Reduce batch size in `run_sinkhorn()` function
- Use a machine with more RAM

### Python Version Issues

This code requires Python 3.6.5 or compatible. If you have version conflicts or lack sudo permissions:
```bash
# Install virtualenv without sudo
python3 -m pip install --user virtualenv

# Create environment with virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Citation

If you use this code for your thesis, please cite the original paper:
```
From Alignment to Assignment: Frustratingly Simple but Effective 
Unsupervised Entity Alignment without Neural Networks
```

## License

Please refer to the original paper's licensing terms.
