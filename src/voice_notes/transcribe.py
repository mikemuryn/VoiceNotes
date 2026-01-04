from __future__ import annotations

import os

# Set Qt to use offscreen platform BEFORE importing whisperx
# whisperx imports pyannote.audio and torchaudio which depend on Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import whisperx


@dataclass
class WhisperResult:
    """Result from Whisper transcription."""
    text: str
    segments: List[Dict[str, Any]]
    language: Optional[str] = None


def transcribe_file(
    audio_path: Path,
    model_name: str,
    device: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> WhisperResult:
    """Transcribe an audio file using WhisperX.

    Args:
        audio_path: Path to the audio/video file.
        model_name: Whisper model name (e.g., "small", "base", "medium").
        device: Device to use ("cpu" or "cuda").
        language: Optional language code (e.g., "en"). If None, will auto-detect.
        prompt: Optional initial prompt for Whisper.

    Returns:
        WhisperResult with text, segments, and detected language.

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError: If model_name or device is invalid.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not model_name:
        raise ValueError("model_name cannot be empty")

    if device not in ("cpu", "cuda"):
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")

    # Prepare ASR options if prompt is provided
    asr_options: Dict[str, Any] = {}
    if prompt:
        asr_options["initial_prompt"] = prompt

    # Load model
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type="float32",
        language=language,
        asr_options=asr_options if asr_options else None,
    )

    # Load audio
    audio = whisperx.load_audio(str(audio_path))

    # Transcribe
    result = model.transcribe(audio, language=language)

    # WhisperX returns a TranscriptionResult object with segments attribute
    # Access segments directly with defensive checks
    segments_raw: List[Any] = []
    if hasattr(result, "segments") and result.segments:
        segments_raw = result.segments if isinstance(result.segments, list) else []

    # Convert segments to list of dicts for consistency
    segments: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    for seg in segments_raw:
        seg_dict: Dict[str, Any]
        if isinstance(seg, dict):
            seg_dict = seg
        else:
            # Convert object to dict with defensive access
            seg_dict = {
                "text": getattr(seg, "text", ""),
                "start": getattr(seg, "start", 0.0),
                "end": getattr(seg, "end", 0.0),
            }
            # Add any other attributes
            if hasattr(seg, "words"):
                words = getattr(seg, "words", [])
                if words:
                    seg_dict["words"] = words
        segments.append(seg_dict)

        # Extract text with defensive access
        seg_text = seg_dict.get("text", "")
        if seg_text and isinstance(seg_text, str):
            seg_text = seg_text.strip()
            if seg_text:
                text_parts.append(seg_text)

    text = " ".join(text_parts).strip()

    # Get detected language with defensive access
    detected_language: Optional[str] = None
    if hasattr(result, "language") and result.language:
        detected_language = result.language
    else:
        detected_language = language  # Fallback to provided language
    
    return WhisperResult(
        text=text,
        segments=segments,
        language=detected_language,
    )


def save_segments_json(segments: List[Dict[str, Any]], path: Path) -> None:
    """Save segments to a JSON file.

    Args:
        segments: List of segment dictionaries.
        path: Path to the output JSON file.

    Raises:
        OSError: If file cannot be written.
        TypeError: If segments cannot be serialized to JSON.
    """
    if not segments:
        segments = []  # Ensure we have a valid list

    try:
        json_content = json.dumps(segments, indent=2, ensure_ascii=False)
        path.write_text(json_content, encoding="utf-8")
    except (TypeError, ValueError) as e:
        raise TypeError(f"Failed to serialize segments to JSON: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to write JSON file {path}: {e}") from e

