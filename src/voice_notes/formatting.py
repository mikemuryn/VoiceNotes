from __future__ import annotations

from typing import Any, Dict, List


def format_speaker_transcript(segments: List[Dict[str, Any]]) -> str:
    """
    Format segments into a speaker-labeled transcript.

    Expected WhisperX segments may include:
      - 'speaker' (e.g., 'SPEAKER_00')
      - 'text'
      - 'start', 'end'

    Returns:
        A readable transcript grouped line-by-line.
    """
    lines: List[str] = []
    for seg in segments:
        speaker = str(seg.get("speaker", "SPEAKER_UNKNOWN"))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines).strip()
