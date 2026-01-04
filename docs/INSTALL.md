# VoiceNotes Installation Guide

This guide provides detailed installation instructions for VoiceNotes on different platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [System Dependencies](#system-dependencies)
- [Installation Methods](#installation-methods)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.11 or higher
- pip or conda package manager
- Git (for cloning the repository)

## System Dependencies

VoiceNotes requires PortAudio, a cross-platform audio I/O library, which is used by `pyannote.audio` and `torchaudio` for audio processing.

### Ubuntu/Debian/WSL

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev libportaudio2
```

**Note:** `portaudio19-dev` provides development headers needed to build Python packages, while `libportaudio2` provides the runtime library.

### macOS

Using Homebrew:

```bash
brew install portaudio
```

If you don't have Homebrew installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install portaudio
```

### Windows

PortAudio is typically included with Python audio packages on Windows. However, if you encounter issues:

1. **Option 1: Use WSL (Recommended)**
   - Install WSL2 and Ubuntu
   - Follow Ubuntu installation instructions above

2. **Option 2: Manual Installation**
   - Download PortAudio from: http://www.portaudio.com/download.html
   - Extract and build according to Windows instructions
   - Add to system PATH

3. **Option 3: Use Conda**
   - Conda packages often include PortAudio:
     ```bash
     conda install -c conda-forge portaudio
     ```

### Verify System Dependencies

After installation, verify PortAudio is available:

**Linux/macOS:**
```bash
pkg-config --modversion portaudio
```

**Or check library:**
```bash
ldconfig -p | grep portaudio  # Linux
brew list portaudio             # macOS
```

## Installation Methods

### Method 1: Using Conda (Recommended)

Conda manages both Python packages and some system dependencies, making it the easiest installation method.

#### Step 1: Install System Dependencies

Follow the [System Dependencies](#system-dependencies) section for your platform.

#### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
```

This creates a new conda environment named `voice` with Python 3.11 and all required packages.

#### Step 3: Activate Environment

```bash
conda activate voice
```

#### Step 4: Install VoiceNotes Package

```bash
pip install -e .
```

For development with all dev dependencies:

```bash
pip install -e ".[dev]"
```

#### Step 5: Verify Installation

```bash
voice-notes --help
```

### Method 2: Using pip with Virtual Environment

#### Step 1: Install System Dependencies

Follow the [System Dependencies](#system-dependencies) section for your platform.

#### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

#### Step 3: Activate Virtual Environment

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

#### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

#### Step 5: Install VoiceNotes

**Basic installation:**
```bash
pip install -e .
```

**With development dependencies:**
```bash
pip install -e ".[dev]"
```

#### Step 6: Verify Installation

```bash
voice-notes --help
```

### Method 3: Using Automated Setup Script

The project includes a `setup.sh` script that automates the installation process:

```bash
chmod +x setup.sh
./setup.sh
```

**Note:** The script will:
- Check for system dependencies and attempt to install them
- Create conda environment (if conda is available) or virtual environment
- Install VoiceNotes in editable mode
- Set up pre-commit hooks

You may need sudo privileges for system dependency installation.

## Verification

After installation, verify everything works:

### 1. Check Command Availability

```bash
voice-notes --help
```

Expected output should show the command-line help.

### 2. Test Basic Functionality

```bash
# This should show help without errors
voice-notes --help

# Check version (if implemented)
voice-notes --version
```

### 3. Test Import

```bash
python -c "import voice_notes; print('Import successful')"
```

### 4. Check Dependencies

```bash
python -c "import torch; import whisperx; import pyannote.audio; print('All dependencies available')"
```

## Troubleshooting

### PortAudio Library Not Found

**Error:**
```
OSError: PortAudio library not found
```

**Solution:**

1. **Verify system dependency installation:**
   ```bash
   # Ubuntu/Debian
   dpkg -l | grep portaudio
   
   # macOS
   brew list portaudio
   ```

2. **Reinstall system dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install --reinstall portaudio19-dev libportaudio2
   
   # macOS
   brew reinstall portaudio
   ```

3. **Reinstall Python audio packages:**
   ```bash
   pip uninstall pyannote.audio torchaudio
   pip install --no-cache-dir pyannote.audio torchaudio
   ```

4. **Set library path (if needed):**
   ```bash
   # Linux
   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   
   # macOS
   export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
   ```

### Command Not Found

**Error:**
```
bash: voice-notes: command not found
```

**Solution:**

1. **Verify environment is activated:**
   ```bash
   which python  # Should point to your venv/conda environment
   ```

2. **Reinstall package:**
   ```bash
   pip install -e .
   ```

3. **Check installation location:**
   ```bash
   pip show voice-notes
   ```

4. **Use Python module directly:**
   ```bash
   python -m voice_notes.cli --help
   ```

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'voice_notes'
```

**Solution:**

1. **Verify you're in the project directory:**
   ```bash
   pwd  # Should be in VoiceNotes directory
   ls src/voice_notes  # Should show package files
   ```

2. **Reinstall in editable mode:**
   ```bash
   pip install -e .
   ```

3. **Check Python path:**
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

### Conda Environment Issues

**Problem:** Conda environment creation fails

**Solution:**

1. **Update conda:**
   ```bash
   conda update conda
   ```

2. **Create environment manually:**
   ```bash
   conda create -n voice python=3.11
   conda activate voice
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Check conda channels:**
   ```bash
   conda config --add channels conda-forge
   ```

### Permission Errors

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**

1. **Don't use sudo with pip** (unless in system Python, which is not recommended)
2. **Use virtual environment or conda** (recommended)
3. **Fix permissions:**
   ```bash
   sudo chown -R $USER:$USER ~/.local
   ```

### Out of Memory During Installation

**Solution:**

1. **Install packages one at a time:**
   ```bash
   pip install torch
   pip install torchaudio
   pip install pyannote.audio
   ```

2. **Use CPU-only PyTorch (smaller):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## Post-Installation Setup

### Pre-commit Hooks (Optional)

For development, install pre-commit hooks:

```bash
pre-commit install
```

### Environment Variables

Set up API keys if using optional features:

```bash
# For speaker diarization
export HUGGINGFACE_TOKEN="your_hf_token"

# For summary generation
export OPENAI_API_KEY="your_openai_key"
```

Add to your `~/.bashrc` or `~/.zshrc` for persistence:

```bash
echo 'export HUGGINGFACE_TOKEN="your_hf_token"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="your_openai_key"' >> ~/.bashrc
```

## Uninstallation

To uninstall VoiceNotes:

```bash
# Deactivate environment
conda deactivate  # or: deactivate

# Remove conda environment
conda env remove -n voice

# Or remove virtual environment
rm -rf venv

# Uninstall package
pip uninstall voice-notes
```

## Getting Help

If you encounter issues not covered here:

1. Check the [README.md](../README.md) for basic usage
2. Review [Troubleshooting](#troubleshooting) section above
3. Check GitHub Issues for similar problems
4. Open a new issue with:
   - Your OS and version
   - Python version (`python --version`)
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](../quickstart.md)
2. Check [Usage Documentation](usage.md) for detailed usage examples
3. Review [Architecture Documentation](../architecture.md) to understand the project structure

