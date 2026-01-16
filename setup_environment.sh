#!/bin/bash

# Setup script for SEU Entity Alignment experiments
# This script sets up the Python environment with conda and downloads required data

echo "=================================================="
echo "SEU Entity Alignment - Environment Setup"
echo "=================================================="

# Check if conda is available
echo -e "\n[1/6] Checking for conda..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH."
    echo "Please install Anaconda or Miniconda from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "conda found: $(conda --version)"

# Create conda environment with Python 3.6.5
echo -e "\n[2/6] Creating conda environment 'seu_env' with Python 3.6.5..."
if conda env list | grep -q "^seu_env "; then
    echo "Conda environment 'seu_env' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n seu_env -y
        conda create -n seu_env python=3.6.5 -y
        echo "Environment recreated successfully."
    fi
else
    conda create -n seu_env python=3.6.5 -y
    echo "Conda environment created successfully."
fi

# Activate conda environment
echo -e "\n[3/6] Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate seu_env

# Install CUDA toolkit and cuDNN via conda
echo -e "\n[4/6] Installing CUDA toolkit 11.2 and cuDNN..."
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

# Install dependencies
echo -e "\n[5/6] Installing Python dependencies..."
pip install --upgrade pip==21.3.1
pip install -r requirements.txt

# Download GloVe embeddings if not present
echo -e "\n[6/6] Checking for GloVe embeddings..."
if [ ! -f "glove.6B.300d.txt" ]; then
    echo "Downloading GloVe embeddings (this may take a while)..."
    if command -v wget &> /dev/null; then
        wget http://nlp.stanford.edu/data/glove.6B.zip
    elif command -v curl &> /dev/null; then
        curl -O http://nlp.stanford.edu/data/glove.6B.zip
    else
        echo "ERROR: Neither wget nor curl is available. Please download manually from:"
        echo "http://nlp.stanford.edu/data/glove.6B.zip"
        exit 1
    fi
    
    echo "Extracting GloVe embeddings..."
    unzip glove.6B.zip glove.6B.300d.txt
    rm glove.6B.zip
    echo "GloVe embeddings downloaded and extracted successfully."
else
    echo "GloVe embeddings already present."
fi

echo -e "\n=================================================="
echo "Setup completed successfully!"
echo "=================================================="
echo -e "\nTo activate the environment in the future, run:"
echo "  conda activate seu_env"
echo -e "\nTo verify GPU setup, run:"
echo "  python -c \"import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))\""
echo -e "\nTo run experiments, use:"
echo "  python run_experiments.py"
echo "  or"
echo "  ./run_all_experiments.sh"
