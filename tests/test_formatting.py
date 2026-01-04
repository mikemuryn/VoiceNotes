"""Tests for formatting module."""

from __future__ import annotations

import pytest

from voice_notes.formatting import format_speaker_transcript


class TestFormatSpeakerTranscript:
    """Test cases for format_speaker_transcript function."""

    def test_valid_segments_with_speakers(self, sample_segments: list[dict]) -> None:
        """Test formatting valid segments with speaker labels."""
        result = format_speaker_transcript(sample_segments)
        expected = (
            "SPEAKER_00: Hello world\n"
            "SPEAKER_01: How are you?\n"
            "SPEAKER_00: I'm doing well, thanks!"
        )
        assert result == expected

    def test_empty_segments_list(self) -> None:
        """Test formatting empty segments list."""
        result = format_speaker_transcript([])
        assert result == ""

    def test_segments_without_speaker_field(self) -> None:
        """Test formatting segments without speaker field."""
        segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.5},
            {"text": "How are you?", "start": 1.5, "end": 3.0},
        ]
        result = format_speaker_transcript(segments)
        expected = "SPEAKER_UNKNOWN: Hello world\nSPEAKER_UNKNOWN: How are you?"
        assert result == expected

    def test_segments_with_empty_text(self) -> None:
        """Test formatting segments with empty text."""
        segments = [
            {"text": "Hello world", "speaker": "SPEAKER_00"},
            {"text": "", "speaker": "SPEAKER_01"},
            {"text": "How are you?", "speaker": "SPEAKER_00"},
        ]
        result = format_speaker_transcript(segments)
        expected = "SPEAKER_00: Hello world\nSPEAKER_00: How are you?"
        assert result == expected

    def test_segments_with_whitespace_only_text(self) -> None:
        """Test formatting segments with whitespace-only text."""
        segments = [
            {"text": "Hello world", "speaker": "SPEAKER_00"},
            {"text": "   \n\t  ", "speaker": "SPEAKER_01"},
            {"text": "How are you?", "speaker": "SPEAKER_00"},
        ]
        result = format_speaker_transcript(segments)
        expected = "SPEAKER_00: Hello world\nSPEAKER_00: How are you?"
        assert result == expected

    def test_segments_with_none_text(self) -> None:
        """Test formatting segments with None text."""
        segments = [
            {"text": "Hello world", "speaker": "SPEAKER_00"},
            {"text": None, "speaker": "SPEAKER_01"},
            {"text": "How are you?", "speaker": "SPEAKER_00"},
        ]
        result = format_speaker_transcript(segments)
        expected = "SPEAKER_00: Hello world\nSPEAKER_00: How are you?"
        assert result == expected

    def test_segments_with_non_string_text(self) -> None:
        """Test formatting segments with non-string text."""
        segments = [
            {"text": "Hello world", "speaker": "SPEAKER_00"},
            {"text": 12345, "speaker": "SPEAKER_01"},
            {"text": "How are you?", "speaker": "SPEAKER_00"},
        ]
        result = format_speaker_transcript(segments)
        expected = "SPEAKER_00: Hello world\nSPEAKER_00: How are you?"
        assert result == expected

    def test_none_input_raises_value_error(self) -> None:
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="segments cannot be None"):
            format_speaker_transcript(None)  # type: ignore[arg-type]

    def test_non_list_input_raises_value_error(self) -> None:
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="segments must be a list"):
            format_speaker_transcript({"not": "a list"})  # type: ignore[arg-type]

    def test_invalid_segment_types_skipped(self) -> None:
        """Test that invalid segment types are skipped."""
        segments = [
            {"text": "Hello world", "speaker": "SPEAKER_00"},
            "not a dict",
            {"text": "How are you?", "speaker": "SPEAKER_00"},
            12345,
            {"text": "Final text", "speaker": "SPEAKER_01"},
        ]
        result = format_speaker_transcript(segments)  # type: ignore[arg-type]
        expected = (
            "SPEAKER_00: Hello world\n"
            "SPEAKER_00: How are you?\n"
            "SPEAKER_01: Final text"
        )
        assert result == expected

    def test_single_segment(self) -> None:
        """Test formatting single segment."""
        segments = [{"text": "Hello world", "speaker": "SPEAKER_00"}]
        result = format_speaker_transcript(segments)
        assert result == "SPEAKER_00: Hello world"

    def test_segments_with_trailing_whitespace(self) -> None:
        """Test that text is stripped of leading/trailing whitespace."""
        segments = [
            {"text": "  Hello world  ", "speaker": "SPEAKER_00"},
            {"text": "\nHow are you?\t", "speaker": "SPEAKER_01"},
        ]
        result = format_speaker_transcript(segments)
        expected = "SPEAKER_00: Hello world\nSPEAKER_01: How are you?"
        assert result == expected

    def test_segments_with_numeric_speaker(self) -> None:
        """Test formatting segments with numeric speaker values."""
        segments = [
            {"text": "Hello world", "speaker": 0},
            {"text": "How are you?", "speaker": 1},
        ]
        result = format_speaker_transcript(segments)
        expected = "0: Hello world\n1: How are you?"
        assert result == expected
