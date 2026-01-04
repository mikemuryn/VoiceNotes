"""Tests for whisperx_tools module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_notes.whisperx_tools import align_transcript, assign_speakers, diarize_audio


class TestAlignTranscript:
    """Test cases for align_transcript function."""

    @patch("voice_notes.whisperx_tools.whisperx.load_audio")
    @patch("voice_notes.whisperx_tools.whisperx.load_align_model")
    @patch("voice_notes.whisperx_tools.whisperx.align")
    def test_successful_alignment(
        self,
        mock_align: MagicMock,
        mock_load_align_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
        sample_aligned_segments: list[dict],
    ) -> None:
        """Test successful alignment with valid inputs."""
        # Setup mocks
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align_model.return_value = (mock_align_model, mock_metadata)
        mock_load_audio.return_value = b"fake_audio_data"
        mock_align.return_value = {"segments": sample_aligned_segments}

        segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.5},
            {"text": "How are you?", "start": 1.5, "end": 3.0},
        ]

        result = align_transcript(
            audio_path=mock_audio_file,
            segments=segments,
            language="en",
            device="cpu",
        )

        assert len(result) == 2
        assert result == sample_aligned_segments
        mock_load_align_model.assert_called_once_with(language_code="en", device="cpu")
        mock_load_audio.assert_called_once_with(str(mock_audio_file))
        mock_align.assert_called_once()

    def test_empty_segments_returns_empty_list(self, mock_audio_file: Path) -> None:
        """Test that empty segments returns empty list."""
        result = align_transcript(
            audio_path=mock_audio_file,
            segments=[],
            language="en",
            device="cpu",
        )
        assert result == []

    def test_file_not_found_raises_error(self, temp_dir: Path) -> None:
        """Test that missing audio file raises FileNotFoundError."""
        non_existent_file = temp_dir / "nonexistent.wav"
        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            align_transcript(
                audio_path=non_existent_file,
                segments=segments,
                language="en",
                device="cpu",
            )

    def test_none_language_raises_error(self, mock_audio_file: Path) -> None:
        """Test that None language raises ValueError."""
        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        with pytest.raises(ValueError, match="Language is required"):
            align_transcript(
                audio_path=mock_audio_file,
                segments=segments,
                language=None,
                device="cpu",
            )

    def test_empty_language_raises_error(self, mock_audio_file: Path) -> None:
        """Test that empty language raises ValueError."""
        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        with pytest.raises(ValueError, match="Language is required"):
            align_transcript(
                audio_path=mock_audio_file,
                segments=segments,
                language="",
                device="cpu",
            )

    def test_whitespace_language_raises_error(self, mock_audio_file: Path) -> None:
        """Test that whitespace-only language raises ValueError."""
        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        with pytest.raises(ValueError, match="Language is required"):
            align_transcript(
                audio_path=mock_audio_file,
                segments=segments,
                language="   ",
                device="cpu",
            )

    def test_invalid_device_raises_error(self, mock_audio_file: Path) -> None:
        """Test that invalid device raises ValueError."""
        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        with pytest.raises(ValueError, match="Invalid device"):
            align_transcript(
                audio_path=mock_audio_file,
                segments=segments,
                language="en",
                device="invalid_device",
            )

    @patch("voice_notes.whisperx_tools.whisperx.load_audio")
    @patch("voice_notes.whisperx_tools.whisperx.load_align_model")
    @patch("voice_notes.whisperx_tools.whisperx.align")
    def test_invalid_alignment_result_raises_runtime_error(
        self,
        mock_align: MagicMock,
        mock_load_align_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test that invalid alignment result raises RuntimeError."""
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align_model.return_value = (mock_align_model, mock_metadata)
        mock_load_audio.return_value = b"fake_audio_data"
        mock_align.return_value = None  # Invalid result

        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        with pytest.raises(RuntimeError, match="Alignment returned invalid result"):
            align_transcript(
                audio_path=mock_audio_file,
                segments=segments,
                language="en",
                device="cpu",
            )

    @patch("voice_notes.whisperx_tools.whisperx.load_audio")
    @patch("voice_notes.whisperx_tools.whisperx.load_align_model")
    @patch("voice_notes.whisperx_tools.whisperx.align")
    def test_alignment_exception_raises_runtime_error(
        self,
        mock_align: MagicMock,
        mock_load_align_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test that alignment exceptions are wrapped in RuntimeError."""
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align_model.return_value = (mock_align_model, mock_metadata)
        mock_load_audio.return_value = b"fake_audio_data"
        mock_align.side_effect = Exception("Alignment failed")

        segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        with pytest.raises(RuntimeError, match="Alignment failed"):
            align_transcript(
                audio_path=mock_audio_file,
                segments=segments,
                language="en",
                device="cpu",
            )


class TestDiarizeAudio:
    """Test cases for diarize_audio function."""

    @patch("voice_notes.whisperx_tools.whisperx.DiarizationPipeline")
    def test_successful_diarization(
        self,
        mock_diarization_pipeline: MagicMock,
        mock_audio_file: Path,
        mock_diarization_result: dict,
    ) -> None:
        """Test successful diarization with valid inputs."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_diarization_result
        mock_diarization_pipeline.return_value = mock_pipeline

        result = diarize_audio(
            audio_path=mock_audio_file,
            device="cpu",
            hf_token="test_token",
            min_speakers=None,
            max_speakers=None,
        )

        assert result == mock_diarization_result
        mock_diarization_pipeline.assert_called_once_with(
            use_auth_token="test_token",
            device="cpu",
        )
        mock_pipeline.assert_called_once_with(
            str(mock_audio_file),
            min_speakers=None,
            max_speakers=None,
        )

    @patch("voice_notes.whisperx_tools.whisperx.DiarizationPipeline")
    def test_diarization_with_speaker_counts(
        self,
        mock_diarization_pipeline: MagicMock,
        mock_audio_file: Path,
        mock_diarization_result: dict,
    ) -> None:
        """Test diarization with speaker count constraints."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_diarization_result
        mock_diarization_pipeline.return_value = mock_pipeline

        result = diarize_audio(
            audio_path=mock_audio_file,
            device="cpu",
            hf_token="test_token",
            min_speakers=2,
            max_speakers=3,
        )

        assert result == mock_diarization_result
        mock_pipeline.assert_called_once_with(
            str(mock_audio_file),
            min_speakers=2,
            max_speakers=3,
        )

    def test_file_not_found_raises_error(self, temp_dir: Path) -> None:
        """Test that missing audio file raises FileNotFoundError."""
        non_existent_file = temp_dir / "nonexistent.wav"
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            diarize_audio(
                audio_path=non_existent_file,
                device="cpu",
                hf_token="test_token",
                min_speakers=None,
                max_speakers=None,
            )

    def test_empty_hf_token_raises_error(self, mock_audio_file: Path) -> None:
        """Test that empty HuggingFace token raises ValueError."""
        with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN is required"):
            diarize_audio(
                audio_path=mock_audio_file,
                device="cpu",
                hf_token="",
                min_speakers=None,
                max_speakers=None,
            )

    def test_whitespace_hf_token_raises_error(self, mock_audio_file: Path) -> None:
        """Test that whitespace-only token raises ValueError."""
        with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN is required"):
            diarize_audio(
                audio_path=mock_audio_file,
                device="cpu",
                hf_token="   ",
                min_speakers=None,
                max_speakers=None,
            )

    def test_invalid_device_raises_error(self, mock_audio_file: Path) -> None:
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            diarize_audio(
                audio_path=mock_audio_file,
                device="invalid_device",
                hf_token="test_token",
                min_speakers=None,
                max_speakers=None,
            )

    def test_min_speakers_less_than_one_raises_error(
        self, mock_audio_file: Path
    ) -> None:
        """Test that min_speakers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_speakers must be at least 1"):
            diarize_audio(
                audio_path=mock_audio_file,
                device="cpu",
                hf_token="test_token",
                min_speakers=0,
                max_speakers=None,
            )

    def test_max_speakers_less_than_one_raises_error(
        self, mock_audio_file: Path
    ) -> None:
        """Test that max_speakers < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_speakers must be at least 1"):
            diarize_audio(
                audio_path=mock_audio_file,
                device="cpu",
                hf_token="test_token",
                min_speakers=None,
                max_speakers=0,
            )

    def test_min_greater_than_max_raises_error(self, mock_audio_file: Path) -> None:
        """Test that min_speakers > max_speakers raises ValueError."""
        with pytest.raises(
            ValueError,
            match="min_speakers cannot be greater than max_speakers",
        ):
            diarize_audio(
                audio_path=mock_audio_file,
                device="cpu",
                hf_token="test_token",
                min_speakers=5,
                max_speakers=2,
            )

    @patch("voice_notes.whisperx_tools.whisperx.DiarizationPipeline")
    def test_diarization_exception_raises_runtime_error(
        self,
        mock_diarization_pipeline: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test that diarization exceptions are wrapped in RuntimeError."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Diarization failed")
        mock_diarization_pipeline.return_value = mock_pipeline

        with pytest.raises(RuntimeError, match="Diarization failed"):
            diarize_audio(
                audio_path=mock_audio_file,
                device="cpu",
                hf_token="test_token",
                min_speakers=None,
                max_speakers=None,
            )


class TestAssignSpeakers:
    """Test cases for assign_speakers function."""

    @patch("voice_notes.whisperx_tools.whisperx.assign_word_speakers")
    def test_successful_speaker_assignment(
        self,
        mock_assign_word_speakers: MagicMock,
        sample_aligned_segments: list[dict],
        mock_diarization_result: dict,
    ) -> None:
        """Test successful speaker assignment."""
        expected_segments = [
            {
                "text": "Hello world",
                "start": 0.0,
                "end": 1.5,
                "speaker": "SPEAKER_00",
            },
            {
                "text": "How are you?",
                "start": 1.5,
                "end": 3.0,
                "speaker": "SPEAKER_01",
            },
        ]
        mock_assign_word_speakers.return_value = {"segments": expected_segments}

        result = assign_speakers(mock_diarization_result, sample_aligned_segments)

        assert result == expected_segments
        mock_assign_word_speakers.assert_called_once()

    def test_empty_segments_returns_empty_list(
        self, mock_diarization_result: dict
    ) -> None:
        """Test that empty segments returns empty list."""
        result = assign_speakers(mock_diarization_result, [])
        assert result == []

    def test_non_list_segments_raises_value_error(
        self, mock_diarization_result: dict
    ) -> None:
        """Test that non-list segments raises ValueError."""
        with pytest.raises(ValueError, match="aligned_segments must be a list"):
            assign_speakers(
                mock_diarization_result, "not a list"  # type: ignore[arg-type]
            )

    @patch("voice_notes.whisperx_tools.whisperx.assign_word_speakers")
    def test_invalid_assignment_result_raises_runtime_error(
        self,
        mock_assign_word_speakers: MagicMock,
        sample_aligned_segments: list[dict],
        mock_diarization_result: dict,
    ) -> None:
        """Test that invalid assignment result raises RuntimeError."""
        mock_assign_word_speakers.return_value = None  # Invalid result

        with pytest.raises(
            RuntimeError, match="Speaker assignment returned invalid result"
        ):
            assign_speakers(mock_diarization_result, sample_aligned_segments)

    @patch("voice_notes.whisperx_tools.whisperx.assign_word_speakers")
    def test_assignment_exception_raises_runtime_error(
        self,
        mock_assign_word_speakers: MagicMock,
        sample_aligned_segments: list[dict],
        mock_diarization_result: dict,
    ) -> None:
        """Test that assignment exceptions are wrapped in RuntimeError."""
        mock_assign_word_speakers.side_effect = Exception("Assignment failed")

        with pytest.raises(RuntimeError, match="Speaker assignment failed"):
            assign_speakers(mock_diarization_result, sample_aligned_segments)

    @patch("voice_notes.whisperx_tools.whisperx.assign_word_speakers")
    def test_value_error_passed_through(
        self,
        mock_assign_word_speakers: MagicMock,
        sample_aligned_segments: list[dict],
        mock_diarization_result: dict,
    ) -> None:
        """Test that ValueError exceptions are passed through."""
        mock_assign_word_speakers.side_effect = ValueError("Invalid input")

        with pytest.raises(ValueError, match="Invalid input"):
            assign_speakers(mock_diarization_result, sample_aligned_segments)
