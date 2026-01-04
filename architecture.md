# VoiceNotes Architecture

## Overview

VoiceNotes transcribes voice recordings using OpenAI Whisper and related technologies.

## Project Structure

```

├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
└── README.md             # Project documentation
```

## Components

### Core Package (`src/voice_notes/`)

Contains:
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

1. Create feature branches for development
2. Format code with Black and isort
3. Run tests with pytest
4. Use pre-commit hooks to check code quality
5. Submit changes via Pull Requests

## Testing Strategy

- Write unit tests for individual components
- Write integration tests for end-to-end workflows
- Maintain coverage at 95% or higher

## Dependencies

- OpenAI Whisper: Core transcription engine
- WhisperX: Enhanced Whisper with alignment
- Faster Whisper: Faster Whisper implementation
- PyAnnote: Speaker diarization
- Rich: CLI formatting

