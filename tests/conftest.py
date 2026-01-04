"""Shared pytest fixtures for VoiceNotes tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for file operations.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_segments() -> list[dict[str, Any]]:
    """Sample segment data for testing.

    Returns:
        List of sample segment dictionaries.
    """
    return [
        {
            "text": "Hello world",
            "start": 0.0,
            "end": 1.5,
            "speaker": "SPEAKER_00",
        },
        {
            "text": "How are you?",
            "start": 1.5,
            "end": 3.0,
            "speaker": "SPEAKER_01",
        },
        {
            "text": "I'm doing well, thanks!",
            "start": 3.0,
            "end": 5.5,
            "speaker": "SPEAKER_00",
        },
    ]


@pytest.fixture
def sample_aligned_segments() -> list[dict[str, Any]]:
    """Sample aligned segments with word-level timestamps.

    Returns:
        List of sample aligned segment dictionaries.
    """
    return [
        {
            "text": "Hello world",
            "start": 0.0,
            "end": 1.5,
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.5},
            ],
        },
        {
            "text": "How are you?",
            "start": 1.5,
            "end": 3.0,
            "words": [
                {"word": "How", "start": 1.5, "end": 1.8},
                {"word": "are", "start": 1.8, "end": 2.2},
                {"word": "you?", "start": 2.2, "end": 3.0},
            ],
        },
    ]


@pytest.fixture
def mock_whisperx_model() -> MagicMock:
    """Mock WhisperX model with predictable responses.

    Returns:
        Mock WhisperX model object.
    """
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.segments = [
        {
            "text": "Hello world",
            "start": 0.0,
            "end": 1.5,
        },
        {
            "text": "How are you?",
            "start": 1.5,
            "end": 3.0,
        },
    ]
    mock_result.language = "en"
    mock_model.transcribe.return_value = mock_result
    return mock_model


@pytest.fixture
def mock_whisperx_transcribe_result() -> dict[str, Any]:
    """Mock WhisperX transcription result.

    Returns:
        Dictionary representing transcription result.
    """
    return {
        "segments": [
            {
                "text": "Hello world",
                "start": 0.0,
                "end": 1.5,
            },
            {
                "text": "How are you?",
                "start": 1.5,
                "end": 3.0,
            },
        ],
        "language": "en",
    }


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client for summarization tests.

    Returns:
        Mock OpenAI client object.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "# Summary\n\nThis is a test summary of the transcript."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_audio_file(temp_dir: Path) -> Path:
    """Create a mock audio file for testing.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Path to mock audio file.
    """
    audio_file = temp_dir / "test_audio.wav"
    # Create a minimal valid WAV file header (44 bytes)
    # This is just enough to pass file existence checks
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)
    return audio_file


@pytest.fixture
def mock_diarization_result() -> dict[str, Any]:
    """Mock diarization result from WhisperX.

    Returns:
        Dictionary representing diarization result.
    """
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "speaker": "SPEAKER_00",
            },
            {
                "start": 1.5,
                "end": 3.0,
                "speaker": "SPEAKER_01",
            },
        ]
    }
