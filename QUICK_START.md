# Master Thesis Experiment Setup - Quick Reference

## Setup on New Machine (One-time)

```bash
# 1. Clone/copy repository to new machine
# 2. Navigate to project directory
cd /path/to/SEU

# 3. Run setup (installs dependencies + downloads GloVe)
./setup_environment.sh

# 4. Verify setup
python verify_setup.py
```

## Running Experiments

### Option 1: Run All Experiments (Recommended for thesis)
```bash
# Runs 3x for all 5 datasets = 15 total experiment runs
./run_all_experiments.sh
```

### Option 2: Test with Single Dataset First
```bash
# Test with one dataset, one run
python run_single_dataset.py dbp_ja_en 1

# Or run multiple times for one dataset
python run_single_dataset.py dbp_ja_en 3
```

### Option 3: Manual Python Execution
```bash
source venv/bin/activate
python run_experiments.py
```

### Option 4: Interactive Jupyter Notebook
```bash
source venv/bin/activate
jupyter notebook main.ipynb
# Run cells manually
```

## Expected Output

All results are saved in `results/` directory:

1. **all_experiments_TIMESTAMP.json** - Complete raw results
   - Individual runs with all metrics
   - Timing information
   - Success/failure status

2. **summary_TIMESTAMP.json** - Statistical summary
   - Mean ± std for each metric
   - Min/max values
   - Aggregated by dataset

3. **experiment_log_TIMESTAMP.txt** - Full execution log
   - Console output
   - Progress updates
   - Any errors/warnings

## Experiment Details

### Datasets (5 total)
- `dbp_fr_en` - DBpedia French-English
- `dbp_ja_en` - DBpedia Japanese-English  
- `dbp_zh_en` - DBpedia Chinese-English
- `srprs_de_en` - SRPRS German-English
- `srprs_fr_en` - SRPRS French-English

### Methods (2 per run)
- **Hungarian Algorithm** - Optimal assignment
- **Sinkhorn Operation** - Iterative normalization

### Metrics Reported
- **Hits@1**: % of correct alignments in top-1 prediction
- **Hits@10**: % of correct alignments in top-10 predictions  
- **MRR**: Mean Reciprocal Rank

### Configuration
- Runs per dataset: **3**
- Feature mode: **hybrid-level** (word + character features)
- Graph depth: **2** (propagation steps)
- Random seed: **12345** (for reproducibility)

## Time Estimates

Per single run (approximate):
- Small datasets (dbp): 3-5 minutes
- Large datasets (srprs): 5-10 minutes

Total for all experiments: **30-60 minutes**

## Troubleshooting

### Setup Issues
```bash
# If environment setup fails
python3 -m pip install --user virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### GloVe Download Issues
```bash
# Manual download if automatic fails
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.300d.txt
rm glove.6B.zip
```

### Memory Issues
- Close other applications
- Run single dataset at a time: `python run_single_dataset.py <dataset> 1`
- Use machine with ≥4GB RAM

### Permission Issues
```bash
chmod +x *.sh *.py
```

## Files Created

```
SEU/
├── requirements.txt              # Python dependencies
├── setup_environment.sh          # Environment setup
├── run_all_experiments.sh        # Master runner
├── run_experiments.py            # Main experiment logic
├── run_single_dataset.py         # Single dataset runner
├── verify_setup.py               # Setup verification
├── SETUP_README.md               # Detailed documentation
└── QUICK_START.md                # This file
```

## For Your Thesis

### Recommended Workflow
1. Set up environment on new machine
2. Run `verify_setup.py` to confirm
3. Do test run: `python run_single_dataset.py dbp_ja_en 1`
4. Run full experiments: `./run_all_experiments.sh`
5. Use results from `results/summary_*.json` for thesis

### Reporting Results
The summary file provides mean ± std for all metrics:
- Report as: "Hits@1: X.XX% ± Y.YY%"
- Based on 3 runs for reproducibility
- Consistent with paper methodology

### Reproducibility Notes
- Fixed random seed (12345)
- 3 runs per dataset (standard practice)
- Same hyperparameters as original paper
- Results saved with timestamps

## Support

If you encounter issues:
1. Check `verify_setup.py` output
2. Review experiment logs in `results/`
3. Ensure GloVe embeddings downloaded correctly
4. Check Python version compatibility (3.6.5+)

Good luck with your master thesis! 🎓
