#!/bin/bash
set -e

echo "XHistoSegMicroSAM Setup"
echo "======================"
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create environment
echo "Creating conda environment 'xhisto' with Python 3.9..."
conda create -n xhisto python=3.9 -y

# Install conda packages
echo "Installing conda packages (python-elf, vigra, nifty)..."
conda install -n xhisto -c conda-forge python-elf vigra nifty -y

# Activate and install pip packages
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate xhisto

echo "Installing Python packages from requirements-py39.txt..."
pip install -r requirements-py39.txt

echo "Installing torch-em from GitHub..."
pip install git+https://github.com/constantinpape/torch-em.git

echo "Installing micro-sam v1.3.0 from GitHub..."
pip install git+https://github.com/computational-cell-analytics/micro-sam.git@v1.3.0

echo ""
echo "Verifying installation..."
if python -c "from xhalo.ml import MicroSAMPredictor; print('✓ Installation successful!')" 2>/dev/null; then
    echo ""
    echo "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Copy .env.example to .env and configure settings:"
    echo "     cp .env.example .env"
    echo ""
    echo "  2. Activate the environment:"
    echo "     conda activate xhisto"
    echo ""
    echo "  3. Run the application:"
    echo "     streamlit run app.py"
    echo ""
    echo "For troubleshooting, see TROUBLESHOOTING.md"
else
    echo "⚠ Warning: Installation verification failed!"
    echo ""
    echo "The import of MicroSAMPredictor failed, which indicates an installation issue."
    echo "This is NOT related to .env configuration and needs to be resolved."
    echo ""
    echo "Common causes:"
    echo "  - Missing dependencies (python-elf, vigra, nifty, torch-em)"
    echo "  - Incorrect Python version (must be 3.9)"
    echo "  - Network issues during package installation"
    echo ""
    echo "Please check TROUBLESHOOTING.md for solutions and try reinstalling."
    exit 1
fi
