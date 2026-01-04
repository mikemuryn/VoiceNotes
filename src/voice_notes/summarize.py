"""Transcript summarization using OpenAI API.

This module provides functionality for generating summaries of transcripts
using OpenAI's chat completion API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class Summary:
    """Summary result from OpenAI API."""
    markdown: str
    text: str


def summarize_transcript(
    transcript: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Summary:
    """Generate a summary of the transcript using OpenAI API.

    Args:
        transcript: The transcript text to summarize.
        model: OpenAI model to use (default: "gpt-4o-mini").
        api_key: OpenAI API key. If None, will try to get from environment.

    Returns:
        Summary object with markdown and text fields.

    Raises:
        ValueError: If API key is not provided, transcript is empty, or model is invalid.
        RuntimeError: If API request fails or response is invalid.
    """
    if not api_key:
        raise ValueError(
            "OpenAI API key is required for summarization. "
            "Set OPENAI_API_KEY environment variable or pass api_key parameter."
        )

    if not transcript or not transcript.strip():
        raise ValueError("transcript cannot be empty")

    if not model or not model.strip():
        raise ValueError("model cannot be empty")

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates concise, well-structured summaries of transcripts. "
                    "Format your response in Markdown with clear sections and bullet points where appropriate.",
                },
                {
                    "role": "user",
                    "content": f"Please summarize the following transcript:\n\n{transcript}",
                },
            ],
        )

        # Defensive access to response structure
        if not response:
            raise RuntimeError("Invalid API response: missing choices")
        
        if not hasattr(response, "choices") or response.choices is None:
            raise RuntimeError("Invalid API response: missing choices")

        if len(response.choices) == 0:
            raise RuntimeError("Invalid API response: empty choices list")

        first_choice = response.choices[0]
        if not first_choice or not first_choice.message:
            raise RuntimeError("Invalid API response: missing message")

        summary_text = first_choice.message.content or ""

        return Summary(
            markdown=summary_text,
            text=summary_text,
        )
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to generate summary: {e}") from e

