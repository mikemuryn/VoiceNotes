"""Tests for CLI module."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_notes.cli import (
    _parse_arguments,
    _process_alignment,
    _process_diarization,
    _process_summary,
    _save_basic_transcript,
    main,
)


class TestParseArguments:
    """Test cases for _parse_arguments function."""

    def test_basic_arguments(self) -> None:
        """Test parsing basic required arguments."""
        with patch.object(sys, "argv", ["voice-notes", "test.wav"]):
            args = _parse_arguments()
            assert args.audio_path == "test.wav"
            assert args.model == "small"
            assert args.device == "cpu"
            assert args.align is False
            assert args.diarize is False
            assert args.summarize is False

    def test_all_arguments(self) -> None:
        """Test parsing all arguments."""
        with patch.object(
            sys,
            "argv",
            [
                "voice-notes",
                "test.wav",
                "--model",
                "base",
                "--device",
                "cuda",
                "--language",
                "en",
                "--prompt",
                "Test prompt",
                "--out",
                "/tmp/output",
                "--align",
                "--diarize",
                "--min-speakers",
                "2",
                "--max-speakers",
                "3",
                "--summarize",
                "--summary-model",
                "gpt-4",
            ],
        ):
            args = _parse_arguments()
            assert args.audio_path == "test.wav"
            assert args.model == "base"
            assert args.device == "cuda"
            assert args.language == "en"
            assert args.prompt == "Test prompt"
            assert args.out == "/tmp/output"
            assert args.align is True
            assert args.diarize is True
            assert args.min_speakers == 2
            assert args.max_speakers == 3
            assert args.summarize is True
            assert args.summary_model == "gpt-4"

    def test_optional_flags(self) -> None:
        """Test parsing optional flags."""
        with patch.object(
            sys, "argv", ["voice-notes", "test.wav", "--align", "--diarize"]
        ):
            args = _parse_arguments()
            assert args.align is True
            assert args.diarize is True
            assert args.summarize is False


class TestSaveBasicTranscript:
    """Test cases for _save_basic_transcript function."""

    @patch("voice_notes.cli.write_text")
    @patch("voice_notes.cli.save_segments_json")
    @patch("voice_notes.cli.console")
    def test_save_basic_transcript(
        self,
        mock_console: MagicMock,
        mock_save_segments: MagicMock,
        mock_write_text: MagicMock,
        temp_dir: Path,
        sample_segments: list[dict],
    ) -> None:
        """Test saving basic transcript."""
        from voice_notes.transcribe import WhisperResult

        whisper_result = WhisperResult(
            text="Hello world",
            segments=sample_segments,
            language="en",
        )

        _save_basic_transcript(whisper_result, temp_dir)

        # Verify files were written
        transcript_path = temp_dir / "transcript.txt"
        segments_path = temp_dir / "segments.json"
        mock_write_text.assert_called_once_with(transcript_path, "Hello world")
        mock_save_segments.assert_called_once_with(sample_segments, segments_path)
        assert mock_console.print.call_count == 2


class TestProcessAlignment:
    """Test cases for _process_alignment function."""

    @patch("voice_notes.cli.align_transcript")
    @patch("voice_notes.cli.save_segments_json")
    @patch("voice_notes.cli.console")
    def test_process_alignment_success(
        self,
        mock_console: MagicMock,
        mock_save_segments: MagicMock,
        mock_align: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
        sample_aligned_segments: list[dict],
    ) -> None:
        """Test successful alignment processing."""
        mock_align.return_value = sample_aligned_segments
        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]

        result = _process_alignment(
            audio_path=mock_audio_file,
            segments=segments,
            language="en",
            device="cpu",
            out_dir=temp_dir,
        )

        assert result == sample_aligned_segments
        mock_align.assert_called_once()
        mock_save_segments.assert_called_once()
        mock_console.print.assert_called_once()

    def test_process_alignment_no_language_raises_error(
        self,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test that missing language raises ValueError."""
        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        with pytest.raises(ValueError, match="Alignment needs a language"):
            _process_alignment(
                audio_path=mock_audio_file,
                segments=segments,
                language=None,
                device="cpu",
                out_dir=temp_dir,
            )


class TestProcessDiarization:
    """Test cases for _process_diarization function."""

    @patch("voice_notes.cli.diarize_audio")
    @patch("voice_notes.cli.assign_speakers")
    @patch("voice_notes.cli.format_speaker_transcript")
    @patch("voice_notes.cli.save_segments_json")
    @patch("voice_notes.cli.write_text")
    @patch("voice_notes.cli.console")
    def test_process_diarization_success(
        self,
        mock_console: MagicMock,
        mock_write_text: MagicMock,
        mock_save_segments: MagicMock,
        mock_format: MagicMock,
        mock_assign: MagicMock,
        mock_diarize: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
        sample_aligned_segments: list[dict],
        mock_diarization_result: dict,
    ) -> None:
        """Test successful diarization processing."""
        mock_diarize.return_value = mock_diarization_result
        mock_assign.return_value = sample_aligned_segments
        mock_format.return_value = "SPEAKER_00: Hello\nSPEAKER_01: World"

        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "test_token"}):
            _process_diarization(
                audio_path=mock_audio_file,
                segments=sample_aligned_segments,
                device="cpu",
                min_speakers=None,
                max_speakers=None,
                out_dir=temp_dir,
            )

        mock_diarize.assert_called_once()
        mock_assign.assert_called_once()
        mock_format.assert_called_once()
        mock_save_segments.assert_called_once()
        mock_write_text.assert_called_once()
        assert mock_console.print.call_count == 2

    def test_process_diarization_no_token_raises_error(
        self,
        mock_audio_file: Path,
        temp_dir: Path,
        sample_aligned_segments: list[dict],
    ) -> None:
        """Test that missing HuggingFace token raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN is required"):
                _process_diarization(
                    audio_path=mock_audio_file,
                    segments=sample_aligned_segments,
                    device="cpu",
                    min_speakers=None,
                    max_speakers=None,
                    out_dir=temp_dir,
                )


class TestProcessSummary:
    """Test cases for _process_summary function."""

    @patch("voice_notes.cli.summarize_transcript")
    @patch("voice_notes.cli.write_text")
    @patch("voice_notes.cli.console")
    def test_process_summary_success(
        self,
        mock_console: MagicMock,
        mock_write_text: MagicMock,
        mock_summarize: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test successful summary processing."""
        from voice_notes.summarize import Summary

        mock_summarize.return_value = Summary(
            markdown="# Summary\n\nTest summary",
            text="# Summary\n\nTest summary",
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            _process_summary(
                transcript="Test transcript",
                model="gpt-4o-mini",
                out_dir=temp_dir,
            )

        mock_summarize.assert_called_once()
        mock_write_text.assert_called_once()
        mock_console.print.assert_called_once()

    def test_process_summary_no_api_key_raises_error(self, temp_dir: Path) -> None:
        """Test that missing OpenAI API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
                _process_summary(
                    transcript="Test transcript",
                    model="gpt-4o-mini",
                    out_dir=temp_dir,
                )


class TestMain:
    """Test cases for main function."""

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli.load_dotenv")
    @patch("voice_notes.cli.console")
    def test_main_basic_transcription(
        self,
        mock_console: MagicMock,
        mock_load_dotenv: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test main function with basic transcription."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="en",
        )

        with patch.object(sys, "argv", ["voice-notes", str(mock_audio_file)]):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                main()

        mock_load_dotenv.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_save_basic.assert_called_once()

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli._process_alignment")
    @patch("voice_notes.cli.load_dotenv")
    def test_main_with_alignment(
        self,
        mock_load_dotenv: MagicMock,
        mock_process_align: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test main function with alignment."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="en",
        )
        mock_process_align.return_value = [
            {"text": "Hello world", "start": 0.0, "end": 1.0}
        ]

        with patch.object(
            sys,
            "argv",
            ["voice-notes", str(mock_audio_file), "--align", "--language", "en"],
        ):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                main()

        mock_process_align.assert_called_once()

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli._process_alignment")
    @patch("voice_notes.cli.load_dotenv")
    def test_main_with_alignment_uses_detected_language(
        self,
        mock_load_dotenv: MagicMock,
        mock_process_align: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test main uses detected language for alignment when --language not set."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="fr",  # Detected language
        )
        mock_process_align.return_value = [
            {"text": "Hello world", "start": 0.0, "end": 1.0}
        ]

        with patch.object(
            sys, "argv", ["voice-notes", str(mock_audio_file), "--align"]
        ):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                main()

        # Verify alignment was called with detected language
        call_args = mock_process_align.call_args
        assert call_args is not None
        assert call_args.kwargs["language"] == "fr"

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli._process_alignment")
    @patch("voice_notes.cli._process_diarization")
    @patch("voice_notes.cli.load_dotenv")
    def test_main_with_diarization(
        self,
        mock_load_dotenv: MagicMock,
        mock_process_diarize: MagicMock,
        mock_process_align: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test main function with diarization."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="en",
        )
        mock_process_align.return_value = [
            {"text": "Hello world", "start": 0.0, "end": 1.0}
        ]

        with patch.object(
            sys,
            "argv",
            [
                "voice-notes",
                str(mock_audio_file),
                "--align",
                "--diarize",
                "--language",
                "en",
            ],
        ):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "test_token"}):
                    main()

        mock_process_diarize.assert_called_once()

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli._process_summary")
    @patch("voice_notes.cli.load_dotenv")
    def test_main_with_summary(
        self,
        mock_load_dotenv: MagicMock,
        mock_process_summary: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test main function with summary."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="en",
        )

        with patch.object(
            sys, "argv", ["voice-notes", str(mock_audio_file), "--summarize"]
        ):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                    main()

        mock_process_summary.assert_called_once()

    @patch("voice_notes.cli.load_dotenv")
    def test_main_file_not_found_raises_error(
        self,
        mock_load_dotenv: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test that missing audio file raises FileNotFoundError."""
        non_existent_file = temp_dir / "nonexistent.wav"
        with patch.object(sys, "argv", ["voice-notes", str(non_existent_file)]):
            with pytest.raises(FileNotFoundError, match="Audio file not found"):
                main()

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli.load_dotenv")
    @patch("voice_notes.cli.console")
    def test_main_with_custom_output_dir(
        self,
        mock_console: MagicMock,
        mock_load_dotenv: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test main function with custom output directory."""
        from voice_notes.transcribe import WhisperResult

        custom_out = temp_dir / "custom_output"
        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="en",
        )

        with patch.object(
            sys, "argv", ["voice-notes", str(mock_audio_file), "--out", str(custom_out)]
        ):
            main()

        assert custom_out.exists()
        mock_save_basic.assert_called_once()

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli.load_dotenv")
    @patch("voice_notes.cli.console")
    def test_main_with_detected_language(
        self,
        mock_console: MagicMock,
        mock_load_dotenv: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test main function prints detected language when available."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="fr",  # Detected language
        )

        with patch.object(sys, "argv", ["voice-notes", str(mock_audio_file)]):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                main()

        # Check that detected language was printed
        language_prints = [
            call
            for call in mock_console.print.call_args_list
            if len(call[0]) > 0 and "Detected language" in str(call[0][0])
        ]
        assert len(language_prints) == 1

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli.load_dotenv")
    @patch("voice_notes.cli.console")
    def test_main_without_detected_language(
        self,
        mock_console: MagicMock,
        mock_load_dotenv: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test main function does not print language when None."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language=None,  # No detected language
        )

        with patch.object(sys, "argv", ["voice-notes", str(mock_audio_file)]):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                main()

        # Check that detected language was NOT printed
        language_prints = [
            call
            for call in mock_console.print.call_args_list
            if len(call[0]) > 0 and "Detected language" in str(call[0][0])
        ]
        assert len(language_prints) == 0

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli._process_summary")
    @patch("voice_notes.cli.load_dotenv")
    @patch("voice_notes.cli.console")
    def test_main_summary_failure_non_fatal(
        self,
        mock_console: MagicMock,
        mock_load_dotenv: MagicMock,
        mock_process_summary: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test that summary failures don't crash the program."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="en",
        )
        mock_process_summary.side_effect = RuntimeError("API quota exceeded")

        with patch.object(
            sys, "argv", ["voice-notes", str(mock_audio_file), "--summarize"]
        ):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                    main()  # Should not raise

        # Verify warning was printed
        warning_prints = [
            call
            for call in mock_console.print.call_args_list
            if len(call[0]) > 0 and "Warning" in str(call[0][0])
        ]
        assert len(warning_prints) >= 1
        mock_save_basic.assert_called_once()  # Basic transcription still completed

    @patch("voice_notes.cli.transcribe_file")
    @patch("voice_notes.cli._save_basic_transcript")
    @patch("voice_notes.cli.load_dotenv")
    @patch("voice_notes.cli.console")
    def test_main_without_summarize_prints_skip_message(
        self,
        mock_console: MagicMock,
        mock_load_dotenv: MagicMock,
        mock_save_basic: MagicMock,
        mock_transcribe: MagicMock,
        mock_audio_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test that main prints skip message when summarize is not requested."""
        from voice_notes.transcribe import WhisperResult

        mock_transcribe.return_value = WhisperResult(
            text="Hello world",
            segments=[{"text": "Hello world", "start": 0.0, "end": 1.0}],
            language="en",
        )

        with patch.object(sys, "argv", ["voice-notes", str(mock_audio_file)]):
            with patch("voice_notes.cli.default_output_dir", return_value=temp_dir):
                main()

        # Check that skip message was printed
        skip_prints = [
            call
            for call in mock_console.print.call_args_list
            if len(call[0]) > 0 and "Skipping summary" in str(call[0][0])
        ]
        assert len(skip_prints) == 1
