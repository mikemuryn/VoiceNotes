#!/bin/bash
# Setup script for VoiceNotes

set -e

echo "Setting up VoiceNotes development environment..."

# Check and install system dependencies
echo "Checking system dependencies..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Check if running on Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo "Detected Ubuntu/Debian system"
        # Check if PortAudio is installed
        if ! dpkg -l | grep -q "portaudio19-dev"; then
            echo "PortAudio not found. Attempting to install..."
            if [ "$EUID" -eq 0 ]; then
                apt-get update
                apt-get install -y portaudio19-dev libportaudio2
            else
                echo "Warning: Need sudo privileges to install system dependencies."
                echo "Please run: sudo apt-get update && sudo apt-get install -y portaudio19-dev libportaudio2"
                echo "Press Enter to continue anyway, or Ctrl+C to abort..."
                read -r
            fi
        else
            echo "PortAudio already installed ✓"
        fi
    else
        echo "Warning: Could not detect package manager. Please install PortAudio manually."
        echo "Ubuntu/Debian: sudo apt-get install portaudio19-dev libportaudio2"
        echo "macOS: brew install portaudio"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        if ! brew list portaudio &> /dev/null; then
            echo "PortAudio not found. Installing via Homebrew..."
            brew install portaudio
        else
            echo "PortAudio already installed ✓"
        fi
    else
        echo "Warning: Homebrew not found. Please install PortAudio manually:"
        echo "  brew install portaudio"
    fi
else
    echo "Warning: Unsupported OS. Please install PortAudio manually."
    echo "See README.md for installation instructions."
fi

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda env create -f environment.yml || echo "Conda environment may already exist, continuing..."
    echo "Activating conda environment..."
    echo "Note: You may need to run 'conda activate voice' manually"
    conda activate voice || echo "Could not activate conda environment automatically. Please run 'conda activate voice' manually."
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
pre-commit install || echo "Pre-commit installation failed. You can install it later with: pre-commit install"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
if command -v conda &> /dev/null; then
    echo "  conda activate voice"
else
    echo "  source venv/bin/activate"
fi
echo ""
echo "Verify installation with:"
echo "  voice-notes --help"

