"""Tests for wrapper module."""

from __future__ import annotations

import os

import pytest


class TestWrapper:
    """Test cases for wrapper module."""

    def test_wrapper_imports_main(self) -> None:
        """Test that wrapper imports main function."""
        import voice_notes.wrapper

        # Verify main is imported
        assert hasattr(voice_notes.wrapper, "main")
        assert voice_notes.wrapper.main is not None

    def test_wrapper_module_has_docstring(self) -> None:
        """Test that wrapper module has docstring."""
        import voice_notes.wrapper

        assert voice_notes.wrapper.__doc__ is not None
        assert "QT_QPA_PLATFORM" in voice_notes.wrapper.__doc__

