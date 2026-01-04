"""Transcript summarization using OpenAI API.

This module provides functionality for generating summaries of transcripts
using OpenAI's chat completion API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openai import APIError, OpenAI, RateLimitError


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
        ValueError: If API key is not provided, transcript is empty,
            or model is invalid.
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
                    "content": """You are summarizing a verbatim transcript
from an audio recording.

First, read the full transcript carefully.

Then produce a concise, structured summary with the sections below.

Use clear language.
Avoid filler.
Do not invent facts.
If something is unclear, note it.

Output format (markdown):

## Summary
- 5–8 bullets capturing the core points

## Decisions
- List explicit decisions made
- If none, write "No explicit decisions recorded"

## Action Items
- Bullet list
- Include owner and due date if stated
- If missing, note "owner not specified" or "no due date stated"

## Open Questions
- Items that were raised but not resolved
- If none, write "No open questions"

## Key Quotes (optional)
- 2–4 short quotes only if they clarify intent or tone

Important rules:
- Base everything strictly on the transcript
- Do not infer beyond what was said
- Preserve intent, not wording""",
                },
                {
                    "role": "user",
                    "content": (f"Summarize the following transcript:\n\n{transcript}"),
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
    except RateLimitError as e:
        error_msg = (
            "OpenAI API quota exceeded. "
            "Please check your billing and usage at https://platform.openai.com/usage. "
            f"Error: {e}"
        )
        raise RuntimeError(error_msg) from e
    except APIError as e:
        error_msg = (
            f"OpenAI API error: {e}. Please check your API key and account status."
        )
        raise RuntimeError(error_msg) from e
    except Exception as e:
        if isinstance(e, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to generate summary: {e}") from e
