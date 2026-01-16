#!/bin/bash

# Setup script for SEU Entity Alignment experiments
# This script sets up the Python environment and downloads required data

echo "=================================================="
echo "SEU Entity Alignment - Environment Setup"
echo "=================================================="

# Check Python version
echo -e "\n[1/5] Checking Python version..."
python3 --version

# Install virtualenv if not available (no sudo needed)
echo -e "\n[2/5] Ensuring virtualenv is available..."
if ! python3 -c "import virtualenv" 2>/dev/null; then
    echo "Installing virtualenv to user directory..."
    python3 -m pip install --user virtualenv
    echo "virtualenv installed successfully."
else
    echo "virtualenv already available."
fi

# Create virtual environment
echo -e "\n[3/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m virtualenv venv
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo -e "\n[4/5] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo -e "\n[5/5] Installing Python dependencies..."
pip install --upgrade pip
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
echo "  source venv/bin/activate"
echo -e "\nTo run experiments, use:"
echo "  python run_experiments.py"
echo "  or"
echo "  ./run_all_experiments.sh"
