from __future__ import annotations

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
    """
    Transcribe an audio file using WhisperX.
    
    Args:
        audio_path: Path to the audio/video file
        model_name: Whisper model name (e.g., "small", "base", "medium")
        device: Device to use ("cpu" or "cuda")
        language: Optional language code (e.g., "en"). If None, will auto-detect.
        prompt: Optional initial prompt for Whisper
        
    Returns:
        WhisperResult with text, segments, and detected language
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Prepare ASR options if prompt is provided
    asr_options = {}
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
    # Access segments directly
    if hasattr(result, "segments"):
        segments_raw = result.segments
    else:
        segments_raw = []
    
    # Convert segments to list of dicts for consistency
    segments = []
    text_parts = []
    for seg in segments_raw:
        if isinstance(seg, dict):
            seg_dict = seg
        else:
            # Convert object to dict
            seg_dict = {
                "text": getattr(seg, "text", ""),
                "start": getattr(seg, "start", 0.0),
                "end": getattr(seg, "end", 0.0),
            }
            # Add any other attributes
            if hasattr(seg, "words"):
                seg_dict["words"] = getattr(seg, "words", [])
        segments.append(seg_dict)
        
        # Extract text
        seg_text = seg_dict.get("text", "").strip()
        if seg_text:
            text_parts.append(seg_text)
    
    text = " ".join(text_parts).strip()
    
    # Get detected language
    if hasattr(result, "language"):
        detected_language = result.language
    else:
        detected_language = language  # Fallback to provided language
    
    return WhisperResult(
        text=text,
        segments=segments,
        language=detected_language,
    )


def save_segments_json(segments: List[Dict[str, Any]], path: Path) -> None:
    """
    Save segments to a JSON file.
    
    Args:
        segments: List of segment dictionaries
        path: Path to the output JSON file
    """
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")

