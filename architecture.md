# VoiceNotes Architecture

## Overview

VoiceNotes is a CLI tool for transcribing voice recordings using OpenAI Whisper and related technologies.

## Project Structure

```
VoiceNotes/
├── src/
│   └── voice_notes/      # Main package
├── apps/                 # Application-specific code
├── deploy/               # Deployment configurations
├── docs/                 # Documentation
├── examples/             # Example scripts and usage
├── scripts/              # Utility scripts
├── standards/            # Coding standards and guidelines
├── tests/                # Test suite
├── pyproject.toml        # Project configuration
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
└── README.md             # Project documentation
```

## Components

### Core Package (`src/voice_notes/`)

The main package containing:
- CLI interface
- Transcription logic
- Whisper integration
- Timestamp alignment
- Speaker diarization

### Configuration

- `pyproject.toml`: Project metadata and build configuration
- `environment.yml`: Conda environment specification
- `requirements.txt`: Python package dependencies
- `setup.cfg`: Flake8 configuration
- `pytest.ini`: Test configuration
- `tox.ini`: Multi-environment testing

## Development Workflow

1. Development happens in feature branches
2. Code is formatted with Black and isort
3. Tests are run with pytest
4. Pre-commit hooks ensure code quality
5. Changes are submitted via Pull Requests

## Testing Strategy

- Unit tests for individual components
- Integration tests for end-to-end workflows
- Coverage target: 80%+

## Dependencies

- OpenAI Whisper: Core transcription engine
- WhisperX: Enhanced Whisper with alignment
- Faster Whisper: Optimized Whisper implementation
- PyAnnote: Speaker diarization
- Rich: CLI formatting

