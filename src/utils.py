"""Utility functions and helpers."""

import os
from typing import Optional


def setup_cudnn_path():
    """Add cuDNN library path for RTX 50 series compatibility."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_dir = os.path.join(current_dir, 'lib')
    
    if os.path.exists(lib_dir) and lib_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = lib_dir + os.pathsep + os.environ.get('PATH', '')


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def clean_subtitle_text(text: str) -> Optional[str]:
    """Remove subtitle artifacts, credits, and watermarks."""
    if not text or not text.strip():
        return None
    
    text = text.strip()
    text_lower = text.lower()
    
    # Common subtitle artifacts to filter out
    artifacts = [
        "translated by",
        "subtitles by", 
        "subtitle by",
        "captioned by",
        "transcribed by",
        "www.",
        "http",
        "subscribe",
        "like and subscribe",
        "follow us",
        "© ",
        "copyright",
        "all rights reserved",
        "opensubtitles",
        "addic7ed",
        "subscene",
        "yify",
        "rarbg"
    ]
    
    # Filter out lines containing artifacts
    for artifact in artifacts:
        if artifact in text_lower:
            return None
    
    # Filter out very short non-meaningful segments
    if len(text) < 3:
        return None
    
    # Filter out lines that are mostly symbols/numbers
    if len([c for c in text if c.isalpha()]) / len(text) < 0.5:
        return None
    
    return text