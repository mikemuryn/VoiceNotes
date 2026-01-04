from __future__ import annotations

from typing import Any, Dict, List


def format_speaker_transcript(segments: List[Dict[str, Any]]) -> str:
    """Format segments into a speaker-labeled transcript.

    Expected WhisperX segments may include:
      - 'speaker' (e.g., 'SPEAKER_00')
      - 'text'
      - 'start', 'end'

    Args:
        segments: List of segment dictionaries with speaker and text information.

    Returns:
        A readable transcript grouped line-by-line.

    Raises:
        ValueError: If segments is None or invalid.
    """
    if segments is None:
        raise ValueError("segments cannot be None")

    if not isinstance(segments, list):
        raise ValueError("segments must be a list")

    lines: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue  # Skip invalid segments

        speaker = str(seg.get("speaker", "SPEAKER_UNKNOWN"))
        text = seg.get("text", "")
        if not text or not isinstance(text, str):
            continue

        text = text.strip()
        if not text:
            continue

        lines.append(f"{speaker}: {text}")

    return "\n".join(lines).strip()
