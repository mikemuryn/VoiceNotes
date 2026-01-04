"""Command-line interface for VoiceNotes.

This module provides the CLI entry point and argument parsing for the
voice-notes application, orchestrating transcription, alignment, diarization,
and summarization workflows.
"""

from __future__ import annotations

import os

# Set Qt to use offscreen platform BEFORE any other imports
# This must be set before importing any Qt-dependent libraries
# (e.g., pyannote.audio, torchaudio)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402

from voice_notes import transcribe  # noqa: E402
from voice_notes.formatting import format_speaker_transcript  # noqa: E402
from voice_notes.io_utils import (  # noqa: E402
    default_output_dir,
    ensure_dir,
    write_text,
)
from voice_notes.summarize import summarize_transcript  # noqa: E402
from voice_notes.transcribe import save_segments_json, transcribe_file  # noqa: E402
from voice_notes.whisperx_tools import (  # noqa: E402
    align_transcript,
    assign_speakers,
    diarize_audio,
)

console = Console()


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.

    Raises:
        SystemExit: If argument parsing fails.
    """
    parser = argparse.ArgumentParser(
        prog="voice-notes",
        description=(
            "Transcribe locally with Whisper; "
            "optionally align and diarize with WhisperX."
        ),
    )
    parser.add_argument("audio_path", type=str, help="Path to audio/video file.")
    parser.add_argument("--model", type=str, default="small", help="Whisper model.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--language", type=str, default=None, help="Language code, e.g. en."
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Initial prompt for Whisper."
    )
    parser.add_argument("--out", type=str, default=None, help="Output directory.")

    parser.add_argument(
        "--align",
        action="store_true",
        help="Run WhisperX alignment for better timestamps.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Run diarization and output transcript_by_speaker.txt.",
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None, help="Minimum speakers (optional)."
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None, help="Maximum speakers (optional)."
    )

    parser.add_argument(
        "--summarize", action="store_true", help="Generate summary.md using API."
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default="gpt-4o-mini",
        help="API model for summary.",
    )
    return parser.parse_args()


def _save_basic_transcript(
    whisper_result: transcribe.WhisperResult,
    out_dir: Path,
) -> None:
    """Save basic transcript and segments.

    Args:
        whisper_result: Transcription result from Whisper.
        out_dir: Output directory path.
    """
    transcript_path = out_dir / "transcript.txt"
    segments_path = out_dir / "segments.json"

    write_text(transcript_path, whisper_result.text)
    save_segments_json(whisper_result.segments, segments_path)

    console.print(f"[green]Wrote:[/green] {transcript_path}")
    console.print(f"[green]Wrote:[/green] {segments_path}")


def _process_alignment(
    audio_path: Path,
    segments: list[dict[str, Any]],
    language: str | None,
    device: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Process WhisperX alignment if requested.

    Args:
        audio_path: Path to audio file.
        segments: Current segments to align.
        language: Language code for alignment.
        device: Device to use for processing.
        out_dir: Output directory path.

    Returns:
        Aligned segments.

    Raises:
        ValueError: If language is required but not provided.
    """
    if not language:
        raise ValueError(
            "Alignment needs a language. Pass --language en or let Whisper detect it."
        )

    aligned_segments = align_transcript(
        audio_path=audio_path,
        segments=segments,
        language=language,
        device=device,
    )
    aligned_path = out_dir / "aligned_segments.json"
    save_segments_json(aligned_segments, aligned_path)
    console.print(f"[green]Wrote:[/green] {aligned_path}")
    return aligned_segments


def _process_diarization(
    audio_path: Path,
    segments: list[dict[str, Any]],
    device: str,
    min_speakers: int | None,
    max_speakers: int | None,
    out_dir: Path,
) -> None:
    """Process speaker diarization if requested.

    Args:
        audio_path: Path to audio file.
        segments: Segments to assign speakers to.
        device: Device to use for processing.
        min_speakers: Minimum number of speakers.
        max_speakers: Maximum number of speakers.
        out_dir: Output directory path.

    Raises:
        ValueError: If HUGGINGFACE_TOKEN is not set.
    """
    hf_token = os.getenv("HUGGINGFACE_TOKEN", "").strip()
    if not hf_token:
        raise ValueError(
            "HUGGINGFACE_TOKEN is required for diarization. "
            "Set it as an environment variable."
        )

    diarization_result = diarize_audio(
        audio_path=audio_path,
        device=device,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    diarized_segments = assign_speakers(diarization_result, segments)

    diarized_path = out_dir / "diarized_segments.json"
    save_segments_json(diarized_segments, diarized_path)
    console.print(f"[green]Wrote:[/green] {diarized_path}")

    by_speaker_text = format_speaker_transcript(diarized_segments)
    by_speaker_path = out_dir / "transcript_by_speaker.txt"
    write_text(by_speaker_path, by_speaker_text)
    console.print(f"[green]Wrote:[/green] {by_speaker_path}")


def _process_summary(
    transcript: str,
    model: str,
    out_dir: Path,
) -> None:
    """Process transcript summary if requested.

    Args:
        transcript: Transcript text to summarize.
        model: OpenAI model to use for summary.
        out_dir: Output directory path.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for summarization. "
            "Set it as an environment variable."
        )

    summary = summarize_transcript(
        transcript=transcript,
        model=model,
        api_key=api_key,
    )
    summary_path = out_dir / "summary.md"
    write_text(summary_path, summary.markdown)
    console.print(f"[green]Wrote:[/green] {summary_path}")


def main() -> None:
    """Entry point for the voice-notes CLI.

    Raises:
        FileNotFoundError: If audio file does not exist.
        ValueError: If required arguments are missing or invalid.
    """
    load_dotenv()

    args = _parse_arguments()

    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else default_output_dir(audio_path)
    )
    ensure_dir(out_dir)

    console.print(f"[bold]Input:[/bold] {audio_path}")
    console.print(f"[bold]Output dir:[/bold] {out_dir}")

    whisper_result = transcribe_file(
        audio_path=audio_path,
        model_name=args.model,
        device=args.device,
        language=args.language,
        prompt=args.prompt,
    )

    _save_basic_transcript(whisper_result, out_dir)

    detected_language = whisper_result.language
    if detected_language:
        console.print(f"[bold]Detected language:[/bold] {detected_language}")

    segments_for_next = whisper_result.segments

    if args.align:
        lang = args.language or detected_language
        segments_for_next = _process_alignment(
            audio_path=audio_path,
            segments=segments_for_next,
            language=lang,
            device=args.device,
            out_dir=out_dir,
        )

    if args.diarize:
        _process_diarization(
            audio_path=audio_path,
            segments=segments_for_next,
            device=args.device,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            out_dir=out_dir,
        )

    if args.summarize:
        try:
            _process_summary(
                transcript=whisper_result.text,
                model=args.summary_model,
                out_dir=out_dir,
            )
        except (ValueError, RuntimeError) as e:
            console.print(f"[yellow]Warning:[/yellow] Summary generation failed: {e}")
            console.print(
                "[yellow]Transcription and alignment completed successfully.[/yellow]"
            )
            # Don't crash - the main work is done
    else:
        console.print("Skipping summary. Run with --summarize to create summary.md.")


if __name__ == "__main__":
    main()
