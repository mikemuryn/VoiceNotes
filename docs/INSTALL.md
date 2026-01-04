# VoiceNotes Installation Guide

Install VoiceNotes on different platforms with these instructions.

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

VoiceNotes requires PortAudio, a cross-platform audio I/O library. The `pyannote.audio` and `torchaudio` packages use it for audio processing.

### Ubuntu/Debian/WSL

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev libportaudio2
```

The `portaudio19-dev` package provides development headers for building Python packages. The `libportaudio2` package provides the runtime library.

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

PortAudio is included with Python audio packages on Windows. If you encounter issues:

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

### Method 1: Using Conda

Conda manages Python packages and some system dependencies. This is the simplest installation method.

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

The script checks for system dependencies and installs them, creates a conda or virtual environment, installs VoiceNotes in editable mode, and sets up pre-commit hooks.

You may need sudo privileges for system dependency installation.

## Verification

After installation, verify everything works:

### 1. Check Command Availability

```bash
voice-notes --help
```

You should see the command-line help.

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

### Qt Platform Plugin Error

If you see errors about Qt platform plugins not being able to initialize:

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" or "wayland"
This application failed to start because no Qt platform plugin could be initialized.
```

This typically occurs in headless environments (WSL, servers, containers). The `voice-notes` command includes a wrapper script that automatically sets `QT_QPA_PLATFORM=offscreen` to prevent this issue.

**Solution:**
- Ensure you're using the installed `voice-notes` command (not running Python modules directly)
- Reinstall the package: `pip install -e .`
- If the issue persists, manually set the environment variable:
  ```bash
  export QT_QPA_PLATFORM=offscreen
  voice-notes your_audio.m4a
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

1. Don't use sudo with pip unless using system Python
2. Use a virtual environment or conda
3. Fix permissions:
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

### Pre-commit Hooks

Install pre-commit hooks for development:

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

1. Check [README.md](../README.md) for basic usage
2. Review the [Troubleshooting](#troubleshooting) section above
3. Check GitHub Issues for similar problems
4. Open a new issue with your OS and version, Python version (`python --version`), full error message, and steps to reproduce

## Next Steps

After installation:

1. Read the [Quick Start Guide](../quickstart.md)
2. Read [Usage Documentation](usage.md) for usage examples
3. Read [Architecture Documentation](../architecture.md) to understand the project structure

