"""Tests for summarize module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_notes.summarize import Summary, summarize_transcript


class TestSummarizeTranscript:
    """Test cases for summarize_transcript function."""

    @patch("voice_notes.summarize.OpenAI")
    def test_successful_summarization(
        self,
        mock_openai_class: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test successful summarization with valid inputs."""
        mock_openai_class.return_value = mock_openai_client

        result = summarize_transcript(
            transcript="This is a test transcript with some content.",
            model="gpt-4o-mini",
            api_key="test_api_key",
        )

        assert isinstance(result, Summary)
        assert result.markdown == "# Summary\n\nThis is a test summary of the transcript."
        assert result.text == "# Summary\n\nThis is a test summary of the transcript."
        mock_openai_class.assert_called_once_with(api_key="test_api_key")
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch("voice_notes.summarize.OpenAI")
    def test_summarization_with_default_model(
        self,
        mock_openai_class: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test summarization with default model."""
        mock_openai_class.return_value = mock_openai_client

        result = summarize_transcript(
            transcript="Test transcript",
            api_key="test_api_key",
        )

        assert isinstance(result, Summary)
        # Verify default model was used
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args is not None
        assert call_args.kwargs["model"] == "gpt-4o-mini"

    @patch("voice_notes.summarize.OpenAI")
    def test_summarization_with_custom_model(
        self,
        mock_openai_class: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test summarization with custom model."""
        mock_openai_class.return_value = mock_openai_client

        result = summarize_transcript(
            transcript="Test transcript",
            model="gpt-4",
            api_key="test_api_key",
        )

        assert isinstance(result, Summary)
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args is not None
        assert call_args.kwargs["model"] == "gpt-4"

    @patch("voice_notes.summarize.OpenAI")
    def test_summarization_with_long_transcript(
        self,
        mock_openai_class: MagicMock,
        mock_openai_client: MagicMock,
    ) -> None:
        """Test summarization with long transcript."""
        mock_openai_class.return_value = mock_openai_client
        long_transcript = "This is a test. " * 1000

        result = summarize_transcript(
            transcript=long_transcript,
            api_key="test_api_key",
        )

        assert isinstance(result, Summary)
        # Verify transcript was included in the request
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args is not None
        assert long_transcript in str(call_args.kwargs["messages"])

    def test_empty_api_key_raises_value_error(self) -> None:
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            summarize_transcript(
                transcript="Test transcript",
                api_key="",
            )

    def test_none_api_key_raises_value_error(self) -> None:
        """Test that None API key raises ValueError."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            summarize_transcript(
                transcript="Test transcript",
                api_key=None,  # type: ignore[arg-type]
            )

    def test_empty_transcript_raises_value_error(self) -> None:
        """Test that empty transcript raises ValueError."""
        with pytest.raises(ValueError, match="transcript cannot be empty"):
            summarize_transcript(
                transcript="",
                api_key="test_api_key",
            )

    def test_whitespace_only_transcript_raises_value_error(self) -> None:
        """Test that whitespace-only transcript raises ValueError."""
        with pytest.raises(ValueError, match="transcript cannot be empty"):
            summarize_transcript(
                transcript="   \n\t  ",
                api_key="test_api_key",
            )

    def test_empty_model_raises_value_error(self) -> None:
        """Test that empty model raises ValueError."""
        with pytest.raises(ValueError, match="model cannot be empty"):
            summarize_transcript(
                transcript="Test transcript",
                model="",
                api_key="test_api_key",
            )

    def test_whitespace_only_model_raises_value_error(self) -> None:
        """Test that whitespace-only model raises ValueError."""
        with pytest.raises(ValueError, match="model cannot be empty"):
            summarize_transcript(
                transcript="Test transcript",
                model="   ",
                api_key="test_api_key",
            )

    @patch("voice_notes.summarize.OpenAI")
    def test_missing_choices_in_response_raises_runtime_error(
        self,
        mock_openai_class: MagicMock,
    ) -> None:
        """Test that missing choices in API response raises RuntimeError."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="Invalid API response: missing choices"):
            summarize_transcript(
                transcript="Test transcript",
                api_key="test_api_key",
            )

    @patch("voice_notes.summarize.OpenAI")
    def test_empty_choices_list_raises_runtime_error(
        self,
        mock_openai_class: MagicMock,
    ) -> None:
        """Test that empty choices list raises RuntimeError."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="Invalid API response: empty choices list"):
            summarize_transcript(
                transcript="Test transcript",
                api_key="test_api_key",
            )

    @patch("voice_notes.summarize.OpenAI")
    def test_missing_message_in_response_raises_runtime_error(
        self,
        mock_openai_class: MagicMock,
    ) -> None:
        """Test that missing message in response raises RuntimeError."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = None
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="Invalid API response: missing message"):
            summarize_transcript(
                transcript="Test transcript",
                api_key="test_api_key",
            )

    @patch("voice_notes.summarize.OpenAI")
    def test_none_content_in_message_returns_empty_summary(
        self,
        mock_openai_class: MagicMock,
    ) -> None:
        """Test that None content in message returns empty summary."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        result = summarize_transcript(
            transcript="Test transcript",
            api_key="test_api_key",
        )

        assert isinstance(result, Summary)
        assert result.markdown == ""
        assert result.text == ""

    @patch("voice_notes.summarize.OpenAI")
    def test_api_exception_raises_runtime_error(
        self,
        mock_openai_class: MagicMock,
    ) -> None:
        """Test that API exceptions are wrapped in RuntimeError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="Failed to generate summary"):
            summarize_transcript(
                transcript="Test transcript",
                api_key="test_api_key",
            )

    @patch("voice_notes.summarize.OpenAI")
    def test_value_error_passed_through(
        self,
        mock_openai_class: MagicMock,
    ) -> None:
        """Test that ValueError exceptions are passed through."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ValueError("Invalid input")
        mock_openai_class.return_value = mock_client

        with pytest.raises(ValueError, match="Invalid input"):
            summarize_transcript(
                transcript="Test transcript",
                api_key="test_api_key",
            )

    @patch("voice_notes.summarize.OpenAI")
    def test_runtime_error_passed_through(
        self,
        mock_openai_class: MagicMock,
    ) -> None:
        """Test that RuntimeError exceptions are passed through."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Runtime error")
        mock_openai_class.return_value = mock_client

        with pytest.raises(RuntimeError, match="Runtime error"):
            summarize_transcript(
                transcript="Test transcript",
                api_key="test_api_key",
            )

