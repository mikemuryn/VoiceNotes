from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console

from voice_notes.formatting import format_speaker_transcript
from voice_notes.io_utils import default_output_dir, ensure_dir, write_text
from voice_notes.summarize import summarize_transcript
from voice_notes.transcribe import save_segments_json, transcribe_file
from voice_notes.whisperx_tools import align_transcript, assign_speakers, diarize_audio

console = Console()


def main() -> None:
    """Entry point for the voice-notes CLI."""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="voice-notes",
        description="Transcribe locally with Whisper; optionally align and diarize with WhisperX.",
    )
    parser.add_argument("audio_path", type=str, help="Path to audio/video file.")
    parser.add_argument("--model", type=str, default="small", help="Whisper model.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--language", type=str, default=None, help="Language code, e.g. en.")
    parser.add_argument("--prompt", type=str, default=None, help="Initial prompt for Whisper.")
    parser.add_argument("--out", type=str, default=None, help="Output directory.")

    parser.add_argument("--align", action="store_true", help="Run WhisperX alignment for better timestamps.")
    parser.add_argument("--diarize", action="store_true", help="Run diarization and output transcript_by_speaker.txt.")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum speakers (optional).")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum speakers (optional).")

    parser.add_argument("--summarize", action="store_true", help="Generate summary.md using API.")
    parser.add_argument("--summary-model", type=str, default="gpt-4o-mini", help="API model for summary.")
    args = parser.parse_args()

    audio_path = Path(args.audio_path).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else default_output_dir(audio_path)
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

    transcript_path = out_dir / "transcript.txt"
    segments_path = out_dir / "segments.json"

    write_text(transcript_path, whisper_result.text)
    save_segments_json(whisper_result.segments, segments_path)

    console.print(f"[green]Wrote:[/green] {transcript_path}")
    console.print(f"[green]Wrote:[/green] {segments_path}")

    detected_language = whisper_result.language
    if detected_language:
        console.print(f"[bold]Detected language:[/bold] {detected_language}")

    segments_for_next = whisper_result.segments

    if args.align:
        if not detected_language and not args.language:
            raise ValueError("Alignment needs a language. Pass --language en or let Whisper detect it.")
        lang = args.language or detected_language
        aligned_segments = align_transcript(
            audio_path=audio_path,
            segments=segments_for_next,
            language=lang,
            device=args.device,
        )
        aligned_path = out_dir / "aligned_segments.json"
        save_segments_json(aligned_segments, aligned_path)
        console.print(f"[green]Wrote:[/green] {aligned_path}")
        segments_for_next = aligned_segments

    if args.diarize:
        hf_token = os.getenv("HUGGINGFACE_TOKEN", "").strip()
        diarization_result = diarize_audio(
            audio_path=audio_path,
            device=args.device,
            hf_token=hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        diarized_segments = assign_speakers(diarization_result, segments_for_next)

        diarized_path = out_dir / "diarized_segments.json"
        save_segments_json(diarized_segments, diarized_path)
        console.print(f"[green]Wrote:[/green] {diarized_path}")

        by_speaker_text = format_speaker_transcript(diarized_segments)
        by_speaker_path = out_dir / "transcript_by_speaker.txt"
        write_text(by_speaker_path, by_speaker_text)
        console.print(f"[green]Wrote:[/green] {by_speaker_path}")

    if args.summarize:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        summary = summarize_transcript(
            transcript=whisper_result.text,
            model=args.summary_model,
            api_key=api_key,
        )
        summary_path = out_dir / "summary.md"
        write_text(summary_path, summary.markdown)
        console.print(f"[green]Wrote:[/green] {summary_path}")
    else:
        console.print("Skipping summary. Run with --summarize to create summary.md.")


if __name__ == "__main__":
    main()
