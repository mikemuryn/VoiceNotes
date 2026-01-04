# VoiceNotes Quick Start Guide

## Installation

### Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate voice
```

### Option 2: Using pip

```bash
pip install -e ".[dev]"
```

**Important:** After installation, verify the command is available:

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

