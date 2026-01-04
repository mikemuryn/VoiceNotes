from __future__ import annotations

from pathlib import Path


def default_output_dir(audio_path: Path) -> Path:
    """Get the default output directory for transcription results.

    Returns the same directory as the input audio file.

    Args:
        audio_path: Path to the audio/video file.

    Returns:
        Path to the output directory.

    Raises:
        ValueError: If audio_path is invalid.
    """
    if not audio_path or str(audio_path) == "":
        raise ValueError("audio_path cannot be None or empty")
    return audio_path.parent


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.

    Raises:
        ValueError: If path is invalid.
        OSError: If directory cannot be created.
    """
    if not path or str(path) == "":
        raise ValueError("path cannot be None or empty")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create directory {path}: {e}") from e


def write_text(path: Path, text: str) -> None:
    """Write text content to a file.

    Args:
        path: Path to the output file.
        text: Text content to write.

    Raises:
        ValueError: If path or text is invalid.
        OSError: If file cannot be written.
    """
    if not path or str(path) == "":
        raise ValueError("path cannot be None or empty")
    if text is None:
        raise ValueError("text cannot be None")
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise OSError(f"Failed to write file {path}: {e}") from e

