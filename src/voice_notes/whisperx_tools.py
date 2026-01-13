"""WhisperX integration for alignment and diarization.

This module provides functions for aligning transcription segments with
word-level timestamps and performing speaker diarization using WhisperX.
"""

from __future__ import annotations

import os

# Set Qt to use offscreen platform BEFORE importing whisperx
# whisperx imports pyannote.audio and torchaudio which depend on Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import whisperx  # noqa: E402


@dataclass(frozen=True)
class WhisperXResult:
    """Outputs from WhisperX alignment and optional diarization."""

    aligned_segments: List[Dict[str, Any]]
    diarized_segments: Optional[List[Dict[str, Any]]]


def align_transcript(
    audio_path: Path,
    segments: List[Dict[str, Any]],
    language: Optional[str],
    device: str,
) -> List[Dict[str, Any]]:
    """Align Whisper segments to get more precise timestamps.

    Args:
        audio_path: Audio/video file path.
        segments: Whisper segments.
        language: Detected or forced language code.
        device: "cpu" or "cuda".

    Returns:
        Aligned segments.

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError: If language is not provided or device is invalid.
        RuntimeError: If alignment fails.
    """
    if not audio_path or not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not language or not language.strip():
        raise ValueError("Language is required for alignment (pass detected language).")

    if device not in ("cpu", "cuda"):
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")

    if not segments:
        return []

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        audio = whisperx.load_audio(str(audio_path))

        aligned = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        if not aligned or not isinstance(aligned, dict):
            raise RuntimeError("Alignment returned invalid result")

        return list(aligned.get("segments", []))
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise RuntimeError(f"Alignment failed: {e}") from e


def diarize_audio(
    audio_path: Path,
    device: str,
    hf_token: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Any:
    """Run diarization to identify speaker turns.

    Args:
        audio_path: Audio/video file path.
        device: Device to use ("cpu" or "cuda").
        hf_token: HuggingFace authentication token.
        min_speakers: Minimum number of speakers (optional).
        max_speakers: Maximum number of speakers (optional).

    Returns:
        A diarization result object compatible with whisperx.assign_word_speakers.

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError: If hf_token is empty or device is invalid.
        RuntimeError: If diarization fails.
    """
    if not audio_path or not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not hf_token or not hf_token.strip():
        raise ValueError("HUGGINGFACE_TOKEN is required for diarization.")

    if device not in ("cpu", "cuda"):
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")

    if min_speakers is not None and min_speakers < 1:
        raise ValueError("min_speakers must be at least 1")

    if max_speakers is not None and max_speakers < 1:
        raise ValueError("max_speakers must be at least 1")

    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise ValueError("min_speakers cannot be greater than max_speakers")

    try:
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )
        result = diarize_model(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return result
    except AttributeError as e:
        # This occurs when Pipeline.from_pretrained() fails and returns None
        # Usually due to authentication/authorization issues
        error_msg = (
            "Failed to load diarization model. This is usually caused by:\n"
            "1. Invalid or missing HUGGINGFACE_TOKEN\n"
            "2. Not accepting the model's terms of use at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Token doesn't have access to the gated model\n\n"
            "To fix:\n"
            "- Get your token from https://huggingface.co/settings/tokens\n"
            "- Accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "- Set HUGGINGFACE_TOKEN environment variable with a valid token"
        )
        raise RuntimeError(error_msg) from e
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        # Check if it's an authentication-related error
        error_str = str(e).lower()
        if any(
            keyword in error_str
            for keyword in ["auth", "token", "unauthorized", "forbidden", "gated"]
        ):
            error_msg = (
                f"Diarization authentication failed: {e}\n\n"
                "This usually means:\n"
                "- Your HUGGINGFACE_TOKEN is invalid or expired\n"
                "- You haven't accepted the model's terms at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "- Your token doesn't have access to the gated model\n\n"
                "Get a new token from https://huggingface.co/settings/tokens"
            )
            raise RuntimeError(error_msg) from e
        raise RuntimeError(f"Diarization failed: {e}") from e


def assign_speakers(
    diarization_result: Any,
    aligned_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign speakers to aligned segments.

    Args:
        diarization_result: Result from diarization pipeline.
        aligned_segments: Aligned segments to assign speakers to.

    Returns:
        Segments annotated with speaker labels where possible.

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If speaker assignment fails.
    """
    if not aligned_segments:
        return []

    if not isinstance(aligned_segments, list):
        raise ValueError("aligned_segments must be a list")

    try:
        assigned = whisperx.assign_word_speakers(
            diarization_result, {"segments": aligned_segments}
        )

        if not assigned or not isinstance(assigned, dict):
            raise RuntimeError("Speaker assignment returned invalid result")

        return list(assigned.get("segments", []))
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Speaker assignment failed: {e}") from e
