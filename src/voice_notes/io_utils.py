from __future__ import annotations

from pathlib import Path


def default_output_dir(audio_path: Path) -> Path:
    """
    Get the default output directory for transcription results.
    
    Returns the same directory as the input audio file.
    
    Args:
        audio_path: Path to the audio/video file
        
    Returns:
        Path to the output directory
    """
    return audio_path.parent


def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
    """
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    """
    Write text content to a file.
    
    Args:
        path: Path to the output file
        text: Text content to write
    """
    path.write_text(text, encoding="utf-8")

