"""Tests for transcribe module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_notes.transcribe import WhisperResult, save_segments_json, transcribe_file


class TestTranscribeFile:
    """Test cases for transcribe_file function."""

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_successful_transcription(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
        sample_segments: list[dict],
    ) -> None:
        """Test successful transcription with valid inputs."""
        # Setup mocks
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.segments = sample_segments
        mock_result.language = "en"
        # Set text to None so code extracts from segments instead
        mock_result.text = None
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        # Call function
        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
            language="en",
        )

        # Verify results
        assert isinstance(result, WhisperResult)
        assert result.language == "en"
        assert len(result.segments) == 3
        assert result.text == "Hello world How are you? I'm doing well, thanks!"

        # Verify mocks were called correctly
        mock_load_model.assert_called_once()
        mock_load_audio.assert_called_once_with(str(mock_audio_file))
        mock_model.transcribe.assert_called_once()

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_transcription_with_prompt(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription with initial prompt."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.segments = [{"text": "Test", "start": 0.0, "end": 1.0}]
        mock_result.language = "en"
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="base",
            device="cpu",
            prompt="This is a test prompt",
        )

        assert isinstance(result, WhisperResult)
        # Verify prompt was passed in asr_options
        call_args = mock_load_model.call_args
        assert call_args is not None
        assert "asr_options" in call_args.kwargs or "asr_options" in str(call_args)

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_transcription_auto_detect_language(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription with auto language detection."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.segments = [{"text": "Bonjour", "start": 0.0, "end": 1.0}]
        mock_result.language = "fr"
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
            language=None,
        )

        assert result.language == "fr"

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_transcription_with_segment_objects(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription when WhisperX returns segment objects instead of dicts."""
        mock_model = MagicMock()
        mock_result = MagicMock()

        # Create mock segment objects
        mock_seg1 = MagicMock()
        mock_seg1.text = "Hello"
        mock_seg1.start = 0.0
        mock_seg1.end = 1.0
        mock_seg1.words = []

        mock_seg2 = MagicMock()
        mock_seg2.text = "World"
        mock_seg2.start = 1.0
        mock_seg2.end = 2.0
        mock_seg2.words = []

        mock_result.segments = [mock_seg1, mock_seg2]
        mock_result.language = "en"
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
        )

        assert len(result.segments) == 2
        assert result.segments[0]["text"] == "Hello"
        assert result.segments[1]["text"] == "World"

    def test_file_not_found_raises_error(self, temp_dir: Path) -> None:
        """Test that missing audio file raises FileNotFoundError."""
        non_existent_file = temp_dir / "nonexistent.wav"
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcribe_file(
                audio_path=non_existent_file,
                model_name="small",
                device="cpu",
            )

    def test_empty_model_name_raises_error(self, mock_audio_file: Path) -> None:
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            transcribe_file(
                audio_path=mock_audio_file,
                model_name="",
                device="cpu",
            )

    def test_invalid_device_raises_error(self, mock_audio_file: Path) -> None:
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            transcribe_file(
                audio_path=mock_audio_file,
                model_name="small",
                device="invalid_device",
            )

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_empty_segments(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription with empty segments."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.segments = []
        mock_result.language = "en"
        # Set text to None so code extracts from segments instead
        mock_result.text = None
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
        )

        assert result.segments == []
        assert result.text == ""

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_segments_with_whitespace_only(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription with segments containing only whitespace."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.segments = [
            {"text": "Hello", "start": 0.0, "end": 1.0},
            {"text": "   ", "start": 1.0, "end": 2.0},
            {"text": "World", "start": 2.0, "end": 3.0},
        ]
        mock_result.language = "en"
        # Set text to None so code extracts from segments instead
        mock_result.text = None
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
        )

        assert result.text == "Hello World"

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_transcription_with_dict_result_and_text(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription when result is a dict with text attribute."""
        mock_model = MagicMock()
        mock_result = {
            "text": "Direct text from result",
            "segments": [{"text": "Hello", "start": 0.0, "end": 1.0}],
            "language": "en",
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
        )

        assert result.text == "Direct text from result"
        assert result.language == "en"

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_transcription_with_dict_result_no_language(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription when result is a dict without language."""
        mock_model = MagicMock()
        mock_result = {
            "segments": [{"text": "Hello", "start": 0.0, "end": 1.0}],
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
            language="fr",
        )

        assert result.language == "fr"  # Falls back to provided language

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_transcription_with_non_list_segments(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription when segments is not a list."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        # Set segments to a tuple (iterable but not list)
        mock_result.segments = ({"text": "Hello", "start": 0.0, "end": 1.0},)
        mock_result.language = "en"
        mock_result.text = None
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
        )

        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "Hello"

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_transcription_with_segments_containing_words(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription when segments have words attribute."""
        mock_model = MagicMock()
        mock_seg = MagicMock()
        mock_seg.text = "Hello"
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.words = [{"word": "Hello", "start": 0.0, "end": 0.5}]

        mock_result = MagicMock()
        mock_result.segments = [mock_seg]
        mock_result.language = "en"
        mock_result.text = None
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
        )

        assert len(result.segments) == 1
        assert "words" in result.segments[0]
        assert result.segments[0]["words"] == [{"word": "Hello", "start": 0.0, "end": 0.5}]

    @patch("voice_notes.transcribe.whisperx.load_audio")
    @patch("voice_notes.transcribe.whisperx.load_model")
    def test_transcription_with_dict_result_non_list_segments(
        self,
        mock_load_model: MagicMock,
        mock_load_audio: MagicMock,
        mock_audio_file: Path,
    ) -> None:
        """Test transcription when dict result has segments that aren't a list."""
        mock_model = MagicMock()
        mock_result = {
            "segments": "not a list",  # Invalid type
        }
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = b"fake_audio_data"

        result = transcribe_file(
            audio_path=mock_audio_file,
            model_name="small",
            device="cpu",
        )

        assert result.segments == []
        assert result.text == ""


class TestSaveSegmentsJson:
    """Test cases for save_segments_json function."""

    def test_save_valid_segments(self, temp_dir: Path, sample_segments: list[dict]) -> None:
        """Test saving valid segments to JSON file."""
        output_path = temp_dir / "segments.json"
        save_segments_json(sample_segments, output_path)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == sample_segments

    def test_save_empty_segments(self, temp_dir: Path) -> None:
        """Test saving empty segments list."""
        output_path = temp_dir / "empty_segments.json"
        save_segments_json([], output_path)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == []

    def test_save_segments_with_unicode(self, temp_dir: Path) -> None:
        """Test saving segments with unicode characters."""
        segments = [
            {"text": "Hello 世界", "start": 0.0, "end": 1.0},
            {"text": "Здравствуй", "start": 1.0, "end": 2.0},
        ]
        output_path = temp_dir / "unicode_segments.json"
        save_segments_json(segments, output_path)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == segments

    def test_save_segments_with_nested_data(self, temp_dir: Path) -> None:
        """Test saving segments with nested data structures."""
        segments = [
            {
                "text": "Hello",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                ],
            },
        ]
        output_path = temp_dir / "nested_segments.json"
        save_segments_json(segments, output_path)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == segments

    def test_overwrite_existing_file(self, temp_dir: Path) -> None:
        """Test overwriting existing JSON file."""
        output_path = temp_dir / "segments.json"
        old_segments = [{"text": "Old", "start": 0.0, "end": 1.0}]
        new_segments = [{"text": "New", "start": 0.0, "end": 1.0}]

        save_segments_json(old_segments, output_path)
        save_segments_json(new_segments, output_path)

        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == new_segments

    def test_save_segments_json_error_handling(self, temp_dir: Path) -> None:
        """Test error handling in save_segments_json."""
        output_path = temp_dir / "segments.json"
        # Create a segment with non-serializable data
        class NonSerializable:
            pass

        segments = [{"text": "Test", "data": NonSerializable()}]

        with pytest.raises(TypeError, match="Failed to serialize segments to JSON"):
            save_segments_json(segments, output_path)  # type: ignore[arg-type]

