"""VoiceNotes - A CLI tool for transcribing voice recordings."""

import os

# Set Qt to use offscreen platform to avoid GUI initialization errors in headless environments
# This must be set before importing any Qt-dependent libraries (e.g., pyannote.audio, torchaudio)
# Set it here in __init__.py so it's applied as soon as the package is imported
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

__version__ = "0.1.0"

