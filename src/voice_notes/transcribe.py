"""Transcription module for audio-to-text conversion using WhisperX.

This module provides functions for transcribing audio files using WhisperX,
including segment extraction and JSON serialization.
"""

from __future__ import annotations

import os

# Set Qt to use offscreen platform BEFORE importing whisperx
# whisperx imports pyannote.audio and torchaudio which depend on Qt
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import whisperx

logger = logging.getLogger(__name__)


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

    # Debug: Log result structure to understand what we're working with
    logger.debug(f"Result type: {type(result)}")
    if isinstance(result, dict):
        logger.debug(f"Result keys: {list(result.keys())}")
        if "segments" in result:
            logger.debug(f"Segments type: {type(result['segments'])}, length: {len(result.get('segments', []))}")
        if "text" in result:
            logger.debug(f"Text available: {bool(result.get('text'))}")
    else:
        logger.debug(f"Result attributes: {dir(result)}")
        if hasattr(result, "segments"):
            logger.debug(f"Segments type: {type(result.segments)}, length: {len(getattr(result, 'segments', []))}")

    # WhisperX can return either a dict or an object
    # Handle both cases with defensive checks
    segments_raw: List[Any] = []
    
    # Try dict access first
    if isinstance(result, dict):
        segments_raw = result.get("segments", [])
        if not isinstance(segments_raw, list):
            segments_raw = []
    # Then try object attribute access
    elif hasattr(result, "segments"):
        segments_value = getattr(result, "segments", None)
        if isinstance(segments_value, list):
            segments_raw = segments_value
        elif segments_value is not None:
            # Convert to list if it's iterable but not a list
            try:
                segments_raw = list(segments_value)
            except (TypeError, ValueError):
                segments_raw = []

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

    # Extract text - try direct access first, then fallback to segments
    text = ""
    
    # Try to get text directly from result (WhisperX often provides this)
    if isinstance(result, dict):
        text = result.get("text", "").strip()
    elif hasattr(result, "text"):
        text_value = getattr(result, "text", None)
        if text_value:
            text = str(text_value).strip()
    
    # Fallback: extract from segments if direct text not available
    if not text and text_parts:
        text = " ".join(text_parts).strip()
    
    logger.debug(f"Extracted text length: {len(text)}, segments count: {len(segments)}")

    # Get detected language with defensive access
    detected_language: Optional[str] = None
    if isinstance(result, dict):
        detected_language = result.get("language") or language
    elif hasattr(result, "language"):
        detected_language = getattr(result, "language", None) or language
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

