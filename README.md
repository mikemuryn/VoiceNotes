# VoiceNotes

Transcribe voice recordings using Whisper and related technologies.

## Features

- Transcribe audio files with OpenAI Whisper
- Align timestamps for word-level accuracy
- Identify different speakers with diarization
- Export transcripts in multiple formats (plain text, JSON, speaker-labeled)

## Installation

See [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

### System Dependencies

Install system dependencies before installing VoiceNotes. These are required by audio processing libraries.

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

After installation, the `voice-notes` command is available in your PATH. If it's not found, activate your virtual or conda environment.

## Usage

### Fast Transcription

```bash
voice-notes recording.m4a --model small --device cpu
```

### Better Timestamps

Get word-level timestamps with WhisperX alignment:

```bash
voice-notes recording.m4a --model small --device cpu --align
```

### Speaker-Labeled Transcript

Identify different speakers with diarization. You need a HuggingFace token:

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

## CPU Usage

When running on CPU:
- Use `base` or `small` models
- Avoid `medium` or `large` models unless needed

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

Reinstall Python audio packages after installing system dependencies:
```bash
pip install --force-reinstall pyannote.audio torchaudio
```

### Command Not Found

If `voice-notes` command is not found:
1. Activate your conda or virtual environment
2. Run `pip install -e .` to verify installation
3. Run with Python: `python -m voice_notes.cli --help`

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

