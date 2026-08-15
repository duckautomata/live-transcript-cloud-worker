"""Transcription provider registry.

Built-in providers register themselves on import; add a new provider module
here (see base.py's docstring for the three-step recipe).
"""

from . import cloudflare, deepgram  # noqa: F401  (import = registration)
from .base import (
    RelativeSegment,
    TranscriptionError,
    TranscriptionProvider,
    get_provider_class,
    register,
    registered_names,
)

__all__ = [
    "RelativeSegment",
    "TranscriptionError",
    "TranscriptionProvider",
    "get_provider_class",
    "register",
    "registered_names",
]
