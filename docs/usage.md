# VoiceNotes Usage Guide

## Command Syntax

```bash
voice-notes <audio_path> [OPTIONS]
```

## Basic Examples

### Fast Transcription

The simplest and fastest way to transcribe an audio file:

```bash
voice-notes recording.m4a --model small --device cpu
```

**Output files:**
- `transcript.txt` - Plain text transcript
- `segments.json` - Basic segments with timestamps

### Better Timestamps

Get word-level timestamps with WhisperX alignment:

```bash
voice-notes recording.m4a --model small --device cpu --align
```

**Output files:**
- `transcript.txt` - Plain text transcript
- `segments.json` - Basic segments
- `aligned_segments.json` - Aligned segments with improved word-level timestamps

### Speaker-Labeled Transcript

Identify different speakers with diarization. Set your HuggingFace token first:

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
```

Then run with diarization:

```bash
voice-notes recording.m4a --model small --device cpu --align --diarize
```

**Output files:**
- `transcript.txt` - Plain text transcript
- `segments.json` - Basic segments
- `aligned_segments.json` - Aligned segments
- `diarized_segments.json` - Segments with speaker assignments
- `transcript_by_speaker.txt` - Speaker-labeled transcript blocks

### Specifying Speaker Count

If you know approximately how many speakers are in the recording, specify the count to improve diarization accuracy:

```bash
voice-notes recording.m4a --align --diarize --min-speakers 2 --max-speakers 3
```

Use this for:
- Interviews (typically 2 speakers)
- Meetings (known number of participants)
- Podcasts (usually 2-4 speakers)

## Command-Line Options

### Required Arguments

- `audio_path` - Path to the audio/video file to transcribe

### Model Options

- `--model` - Whisper model to use (default: `small`)
  - Options: `tiny`, `base`, `small`, `medium`, `large`
  - For CPU: use `base` or `small`
  - For GPU: can use `medium` or `large`

- `--device` - Device to use for processing (default: `cpu`)
  - Options: `cpu`, `cuda`
  - Use `cuda` if you have a compatible GPU

### Language Options

- `--language` - Language code (e.g., `en`, `es`, `fr`)
  - Optional: Whisper will auto-detect if not specified
  - Required when using `--align` if auto-detection fails

- `--prompt` - Initial prompt for Whisper
  - Improves accuracy for specific domains or terminology

### Processing Options

- `--align` - Run WhisperX alignment for word-level timestamps
  - Requires language detection or `--language` flag

- `--diarize` - Run speaker diarization
  - Identifies different speakers in the audio
  - Requires `HUGGINGFACE_TOKEN` environment variable
  - Works best with `--align` enabled

- `--min-speakers` - Minimum number of speakers (optional)
  - Improves diarization accuracy

- `--max-speakers` - Maximum number of speakers (optional)
  - Improves diarization accuracy

### Output Options

- `--out` - Output directory (optional)
  - Default: same directory as input file

- `--summarize` - Generate summary.md using OpenAI API
  - Requires `OPENAI_API_KEY` environment variable
  - Uses GPT-4o-mini by default

- `--summary-model` - API model for summary (default: `gpt-4o-mini`)

## Environment Variables

### HUGGINGFACE_TOKEN

Required for speaker diarization. Get your token from [HuggingFace](https://huggingface.co/settings/tokens).

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
```

### OPENAI_API_KEY

Required for summary generation. Get your API key from [OpenAI](https://platform.openai.com/api-keys).

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## Output Files

### transcript.txt
Plain text transcript of the audio, without timestamps or speaker labels.

### segments.json
JSON file containing segments with start/end times and text. Format:
```json
[
  {
    "start": 0.0,
    "end": 5.2,
    "text": "Hello, this is a test."
  }
]
```

### aligned_segments.json
Similar to `segments.json` but with improved word-level timestamps from WhisperX alignment.

### diarized_segments.json
Segments with speaker assignments. Format:
```json
[
  {
    "start": 0.0,
    "end": 5.2,
    "text": "Hello, this is a test.",
    "speaker": "SPEAKER_00"
  }
]
```

### transcript_by_speaker.txt
Human-readable transcript organized by speaker, with clear speaker labels.

### summary.md
Markdown summary of the transcript (only generated with `--summarize` flag).

## Performance Tips

1. CPU Usage: Use `--device cpu` with `base` or `small` models
2. GPU Usage: Use `--device cuda` with larger models for faster processing
3. Model Selection:
   - `tiny` or `base` - Fastest, use for quick transcriptions
   - `small` - Balanced speed and accuracy
   - `medium` or `large` - Best accuracy, slower (use with GPU)
4. Diarization: Use only when needed. It increases processing time.
5. Speaker Count: Use `--min-speakers` and `--max-speakers` to improve diarization accuracy

## Common Use Cases

### Quick Meeting Notes
```bash
voice-notes meeting.m4a --model small --device cpu
```

### Detailed Interview Transcription
```bash
export HUGGINGFACE_TOKEN="your_token"
voice-notes interview.m4a --model small --device cpu --align --diarize --min-speakers 2 --max-speakers 2
```

### Podcast Transcription with Summary
```bash
export HUGGINGFACE_TOKEN="your_hf_token"
export OPENAI_API_KEY="your_openai_key"
voice-notes podcast.m4a --model small --device cpu --align --diarize --summarize
```

