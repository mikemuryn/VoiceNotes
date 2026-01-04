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
    """
    Generate a summary of the transcript using OpenAI API.
    
    Args:
        transcript: The transcript text to summarize
        model: OpenAI model to use (default: "gpt-4o-mini")
        api_key: OpenAI API key. If None, will try to get from environment.
        
    Returns:
        Summary object with markdown and text fields
        
    Raises:
        ValueError: If API key is not provided and not in environment
    """
    if not api_key:
        raise ValueError(
            "OpenAI API key is required for summarization. "
            "Set OPENAI_API_KEY environment variable or pass api_key parameter."
        )
    
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
    
    summary_text = response.choices[0].message.content or ""
    
    return Summary(
        markdown=summary_text,
        text=summary_text,
    )

