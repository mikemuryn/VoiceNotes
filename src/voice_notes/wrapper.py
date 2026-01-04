#!/usr/bin/env python3
"""Wrapper script for voice-notes CLI.

Sets QT_QPA_PLATFORM=offscreen before any imports to avoid Qt GUI errors
in headless environments (WSL, servers, containers).
"""
import os

# CRITICAL: Set this BEFORE any other imports, even __future__
# This ensures Qt libraries see the environment variable during initialization
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from voice_notes.cli import main

if __name__ == "__main__":
    main()

