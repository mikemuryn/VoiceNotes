from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import whisperx


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
    """
    Align Whisper segments to get more precise timestamps (and often better segmentation).

    Args:
        audio_path: Audio/video file path.
        segments: Whisper segments.
        language: Detected or forced language code.
        device: "cpu" or "cuda".

    Returns:
        Aligned segments.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not language:
        raise ValueError("Language is required for alignment (pass detected language).")

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    audio = whisperx.load_audio(str(audio_path))

    aligned = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    return list(aligned.get("segments", []))


def diarize_audio(
    audio_path: Path,
    device: str,
    hf_token: str,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Any:
    """
    Run diarization to identify speaker turns.

    Returns:
        A diarization result object compatible with whisperx.assign_word_speakers.
    """
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN is required for diarization.")

    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token,
        device=device,
    )
    return diarize_model(
        str(audio_path),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def assign_speakers(
    diarization_result: Any,
    aligned_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Assign speakers to aligned segments.

    Returns:
        Segments annotated with speaker labels where possible.
    """
    assigned = whisperx.assign_word_speakers(diarization_result, {"segments": aligned_segments})
    return list(assigned.get("segments", []))
