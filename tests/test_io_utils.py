"""Tests for io_utils module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voice_notes.io_utils import default_output_dir, ensure_dir, write_text


class TestDefaultOutputDir:
    """Test cases for default_output_dir function."""

    def test_valid_path(self, temp_dir: Path) -> None:
        """Test default output dir for valid path."""
        audio_file = temp_dir / "test_audio.wav"
        audio_file.touch()
        result = default_output_dir(audio_file)
        assert result == temp_dir

    def test_nested_path(self, temp_dir: Path) -> None:
        """Test default output dir for nested path."""
        nested_dir = temp_dir / "subdir" / "nested"
        nested_dir.mkdir(parents=True)
        audio_file = nested_dir / "test_audio.wav"
        audio_file.touch()
        result = default_output_dir(audio_file)
        assert result == nested_dir

    def test_path_with_dots(self, temp_dir: Path) -> None:
        """Test default output dir for path with dots."""
        audio_file = temp_dir / "test.audio.file.wav"
        audio_file.touch()
        result = default_output_dir(audio_file)
        assert result == temp_dir

    def test_none_path_raises_value_error(self) -> None:
        """Test that None path raises ValueError."""
        with pytest.raises(ValueError, match="audio_path cannot be None or empty"):
            default_output_dir(None)  # type: ignore[arg-type]

    def test_empty_path_raises_value_error(self) -> None:
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="audio_path cannot be None or empty"):
            default_output_dir(Path(""))


class TestEnsureDir:
    """Test cases for ensure_dir function."""

    def test_create_new_directory(self, temp_dir: Path) -> None:
        """Test creating a new directory."""
        new_dir = temp_dir / "new_dir"
        ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_existing_directory(self, temp_dir: Path) -> None:
        """Test with existing directory."""
        existing_dir = temp_dir / "existing_dir"
        existing_dir.mkdir()
        ensure_dir(existing_dir)
        assert existing_dir.exists()
        assert existing_dir.is_dir()

    def test_nested_paths(self, temp_dir: Path) -> None:
        """Test creating nested directory paths."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        ensure_dir(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert (temp_dir / "level1").exists()
        assert (temp_dir / "level1" / "level2").exists()

    def test_none_path_raises_value_error(self) -> None:
        """Test that None path raises ValueError."""
        with pytest.raises(ValueError, match="path cannot be None or empty"):
            ensure_dir(None)  # type: ignore[arg-type]

    def test_empty_path_raises_value_error(self) -> None:
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="path cannot be None or empty"):
            ensure_dir(Path(""))


class TestWriteText:
    """Test cases for write_text function."""

    def test_normal_write(self, temp_dir: Path) -> None:
        """Test writing normal text to file."""
        file_path = temp_dir / "test.txt"
        content = "Hello, world!"
        write_text(file_path, content)
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == content

    def test_empty_text(self, temp_dir: Path) -> None:
        """Test writing empty text."""
        file_path = temp_dir / "empty.txt"
        write_text(file_path, "")
        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == ""

    def test_multiline_text(self, temp_dir: Path) -> None:
        """Test writing multiline text."""
        file_path = temp_dir / "multiline.txt"
        content = "Line 1\nLine 2\nLine 3"
        write_text(file_path, content)
        assert file_path.read_text(encoding="utf-8") == content

    def test_unicode_text(self, temp_dir: Path) -> None:
        """Test writing unicode text."""
        file_path = temp_dir / "unicode.txt"
        content = "Hello 世界 🌍 Здравствуй"
        write_text(file_path, content)
        assert file_path.read_text(encoding="utf-8") == content

    def test_overwrite_existing_file(self, temp_dir: Path) -> None:
        """Test overwriting existing file."""
        file_path = temp_dir / "existing.txt"
        file_path.write_text("Old content", encoding="utf-8")
        write_text(file_path, "New content")
        assert file_path.read_text(encoding="utf-8") == "New content"

    def test_none_path_raises_value_error(self) -> None:
        """Test that None path raises ValueError."""
        with pytest.raises(ValueError, match="path cannot be None or empty"):
            write_text(None, "content")  # type: ignore[arg-type]

    def test_empty_path_raises_value_error(self) -> None:
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="path cannot be None or empty"):
            write_text(Path(""), "content")

    def test_none_text_raises_value_error(self, temp_dir: Path) -> None:
        """Test that None text raises ValueError."""
        file_path = temp_dir / "test.txt"
        with pytest.raises(ValueError, match="text cannot be None"):
            write_text(file_path, None)  # type: ignore[arg-type]

    def test_special_characters(self, temp_dir: Path) -> None:
        """Test writing text with special characters."""
        file_path = temp_dir / "special.txt"
        content = "Special chars: !@#$%^&*()[]{}|\\/<>?~`"
        write_text(file_path, content)
        assert file_path.read_text(encoding="utf-8") == content

    def test_very_long_text(self, temp_dir: Path) -> None:
        """Test writing very long text."""
        file_path = temp_dir / "long.txt"
        content = "A" * 10000
        write_text(file_path, content)
        assert file_path.read_text(encoding="utf-8") == content
        assert len(file_path.read_text(encoding="utf-8")) == 10000

