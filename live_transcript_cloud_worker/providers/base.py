"""Transcription provider plugin interface.

Adding a new third-party transcription API is one self-contained module:

1. Create ``providers/<name>.py`` with a ``TranscriptionProvider`` subclass:

       from .base import RelativeSegment, TranscriptionError, TranscriptionProvider, register

       @register
       class AcmeProvider(TranscriptionProvider):
           name = "acme"
           known_options = frozenset({"api_key", "model"})
           env_overrides = {"api_key": "ACME_API_KEY"}

           @classmethod
           def validate_options(cls, options):
               return [] if options.get("api_key") else ["needs api_key (or ACME_API_KEY)"]

           async def transcribe(self, wav: bytes) -> list[RelativeSegment]:
               ...POST wav via self.client, self.options, self.language...
               # raise TranscriptionError(msg, retryable=...) on failure;
               # return [] for silence, that is a result, not an error.

2. Import it in ``providers/__init__.py`` so registration runs.
3. Configure it: ``transcription.provider: acme`` plus a ``transcription.acme:``
   options block in the YAML. Config validation, env overrides, retries, and
   fallback all come for free.

Contract for ``transcribe``:
- input is a complete WAV file (16 kHz mono s16le, header included);
- return segments relative to the chunk start: ``[(start_seconds, text), ...]``;
- an empty list is a legitimate result (silence) and is never retried;
- every failure must raise ``TranscriptionError`` with ``retryable`` set
  honestly, the service retries retryable errors with backoff and then moves
  to the fallback provider. Unexpected exceptions are treated as retryable,
  but classify what you can.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import httpx

RelativeSegment = tuple[float, str]


class TranscriptionError(Exception):
    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class TranscriptionProvider(ABC):
    #: registry key; also the YAML section name under ``transcription:``
    name: ClassVar[str]
    #: every YAML option key this provider accepts (typo protection)
    known_options: ClassVar[frozenset[str]] = frozenset()
    #: option key -> environment variable that overrides it (env wins)
    env_overrides: ClassVar[dict[str, str]] = {}

    def __init__(self, options: dict[str, Any], language: str, client: httpx.AsyncClient) -> None:
        self.options = options
        self.language = language
        self.client = client

    @classmethod
    def validate(cls, options: dict[str, Any]) -> list[str]:
        """Config-time validation; returns human-readable problems."""
        problems = []
        unknown = set(options) - cls.known_options
        if unknown:
            problems.append(f"unknown option(s): {', '.join(sorted(unknown))}")
        problems.extend(cls.validate_options(options))
        return problems

    @classmethod
    def validate_options(cls, options: dict[str, Any]) -> list[str]:
        """Provider-specific checks (required credentials etc.)."""
        return []

    @abstractmethod
    async def transcribe(self, wav: bytes) -> list[RelativeSegment]:
        """Transcribe one WAV chunk. See the module docstring for the contract."""


_REGISTRY: dict[str, type[TranscriptionProvider]] = {}


def register(cls: type[TranscriptionProvider]) -> type[TranscriptionProvider]:
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls:
        # A copied provider module with an unchanged `name` would silently
        # hijack the original's registry slot; fail at import instead.
        raise ValueError(f"duplicate provider name {cls.name!r}: {existing.__qualname__} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_provider_class(name: str) -> type[TranscriptionProvider] | None:
    return _REGISTRY.get(name)


def registered_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
