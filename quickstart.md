# VoiceNotes Quick Start Guide

## Installation

### Step 1: Install System Dependencies

VoiceNotes requires PortAudio for audio processing. Install it based on your system:

**Ubuntu/Debian/WSL:**
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev libportaudio2
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
PortAudio is typically included with Python audio packages. If you encounter issues, use WSL or install manually.

### Step 2: Install Python Package

#### Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate voice
pip install -e .
```

#### Option 2: Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Step 3: Verify Installation

```bash
voice-notes --help
```

If the command is not found, make sure:
1. Your virtual/conda environment is activated
2. The environment's bin directory is in your PATH
3. You've run `pip install -e .` (or `pip install -e ".[dev]"`) in the project directory

## Basic Usage

### 1. Fast Transcription

Transcribe an audio file with the fastest model:

```bash
voice-notes recording.m4a --model small --device cpu
```

This creates `transcript.txt` with the plain text transcription.

### 2. Transcription with Better Timestamps

Get improved timestamp accuracy using WhisperX alignment:

```bash
voice-notes recording.m4a --model small --device cpu --align
```

This creates:
- `transcript.txt` - Plain transcript
- `segments.json` - Timestamped segments
- `aligned_segments.json` - Aligned segments with better timestamps

### 3. Transcription with Speaker Labels

Identify different speakers (requires HuggingFace token):

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
voice-notes recording.m4a --model small --device cpu --align --diarize
```

This creates:
- `transcript.txt` - Plain transcript
- `segments.json` - Timestamped segments
- `aligned_segments.json` - Aligned segments
- `diarized_segments.json` - Segments with speaker assignments
- `transcript_by_speaker.txt` - Speaker-labeled blocks

### 4. Specifying Speaker Count

If you know approximately how many speakers are in the recording:

```bash
voice-notes recording.m4a --align --diarize --min-speakers 2 --max-speakers 3
```

This helps improve diarization accuracy when you have a known speaker count.

## Model Selection

Choose the right model for your needs:

- `tiny` - Fastest, least accurate
- `base` - Good balance (recommended for CPU)
- `small` - Better accuracy, still fast
- `medium` - High accuracy, slower
- `large` - Best accuracy, slowest (GPU recommended)

## Tips

- For CPU usage, stick with `base` or `small` models and use `--device cpu`
- Use `--align` for better timestamp accuracy
- Use `--diarize` only when you need speaker identification (it's slower)
- Set `HUGGINGFACE_TOKEN` environment variable before running diarization
- Use `--min-speakers` and `--max-speakers` to improve diarization accuracy when you know the speaker count
- Output files are created in the same directory as the input file

## Troubleshooting

### PortAudio Library Not Found

If you see `OSError: PortAudio library not found`:

1. **Install system dependencies** (see Step 1 above)
2. **Reinstall audio packages:**
   ```bash
   pip install --force-reinstall pyannote.audio torchaudio
   ```

### Command Not Found

If `voice-notes` command is not found:
1. Activate your conda/virtual environment
2. Run `pip install -e .` in the project directory
3. Verify with `voice-notes --help`
4. If still not found, try: `python -m voice_notes.cli --help`

### Out of Memory

- Use a smaller model (`tiny`, `base`, or `small`)
- Process shorter audio files
- Close other applications

### Slow Performance

- Use `faster-whisper` backend (default)
- Choose a smaller model
- Consider using GPU if available

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
- See [examples/](examples/) for more usage examples

