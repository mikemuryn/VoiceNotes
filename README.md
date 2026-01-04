# VoiceNotes

A CLI tool for transcribing voice recordings using Whisper and related technologies.

## Features

- Fast transcription using OpenAI Whisper
- Timestamp alignment for better accuracy
- Speaker diarization support
- Multiple output formats (plain text, JSON, speaker-labeled)

## Installation

### Using Conda

```bash
conda env create -f environment.yml
conda activate voice
```

### Using pip

```bash
pip install -e ".[dev]"
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

