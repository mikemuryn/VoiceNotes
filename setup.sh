#!/bin/bash
# Setup script for VoiceNotes

set -e

echo "Setting up VoiceNotes development environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda env create -f environment.yml
    echo "Activating conda environment..."
    conda activate voice
else
    echo "Conda not found. Using pip instead..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
fi

# Install package in editable mode
echo "Installing VoiceNotes in editable mode..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo "Setup complete!"
echo ""
echo "To activate the environment:"
if command -v conda &> /dev/null; then
    echo "  conda activate voice"
else
    echo "  source venv/bin/activate"
fi

