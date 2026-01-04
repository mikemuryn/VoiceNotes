# VoiceNotes

A CLI tool for transcribing voice recordings using Whisper and related technologies.

## Features

- Fast transcription using OpenAI Whisper
- Timestamp alignment for better accuracy
- Speaker diarization support
- Multiple output formats (plain text, JSON, speaker-labeled)

## Installation

> **For detailed installation instructions, see [docs/INSTALL.md](docs/INSTALL.md)**

### System Dependencies

Before installing VoiceNotes, you need to install system dependencies required by audio processing libraries.

#### Ubuntu/Debian/WSL

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev libportaudio2
```

#### macOS

```bash
brew install portaudio
```

#### Windows

PortAudio is typically included with Python audio packages on Windows. If you encounter issues, you may need to install it manually or use WSL.

### Using Conda (Recommended)

1. **Install system dependencies** (see above)

2. **Create and activate conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate voice
   ```

3. **Install VoiceNotes package:**
   ```bash
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   voice-notes --help
   ```

### Using pip

1. **Install system dependencies** (see above)

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install VoiceNotes:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation:**
   ```bash
   voice-notes --help
   ```

**Note:** After installation, the `voice-notes` command will be available in your PATH. If it's not found, make sure your Python environment's bin directory is in your PATH, or activate your virtual/conda environment.

## Usage

### Fast Transcription

```bash
voice-notes recording.m4a --model small --device cpu
```

### Better Timestamps

For improved timestamp accuracy using WhisperX alignment:

```bash
voice-notes recording.m4a --model small --device cpu --align
```

### Speaker-Labeled Transcript (slowest)

For speaker diarization, you'll need a HuggingFace token:

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
voice-notes recording.m4a --model small --device cpu --align --diarize
```

### Specifying Speaker Count

If you roughly know how many speakers are in the recording:

```bash
voice-notes recording.m4a --align --diarize --min-speakers 2 --max-speakers 3
```

## Output Files

- `transcript.txt` - Plain transcript
- `segments.json` - Timestamps
- `transcript_by_speaker.txt` - Speaker-labeled blocks (only with diarization)
- `summary.md` - Optional summary (if API key is provided)

## CPU-Specific Tips

On CPU, choose models like:
- `base` or `small` for daily use
- Avoid `medium`/`large` unless you really need it

## Troubleshooting

### PortAudio Library Not Found

If you see `OSError: PortAudio library not found`, you need to install system dependencies:

**Ubuntu/Debian/WSL:**
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev libportaudio2
```

**macOS:**
```bash
brew install portaudio
```

After installing, you may need to reinstall Python audio packages:
```bash
pip install --force-reinstall pyannote.audio torchaudio
```

### Command Not Found

If `voice-notes` command is not found:
1. Make sure your conda/virtual environment is activated
2. Verify installation: `pip install -e .`
3. Check PATH includes your environment's bin directory
4. Try running with Python: `python -m voice_notes.cli --help`

## Development

### Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

## License

MIT License - see LICENSE file for details.

