"""Transcription service: a configured provider with bounded retries, then
an optional fallback provider.

Provider implementations live in the ``providers`` package and register
themselves by name, see providers/base.py for how to add a new API. This
module only owns the retry/fallback policy around whichever providers the
config selects.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from .config import TranscriptionConfig
from .providers import RelativeSegment, TranscriptionError, get_provider_class

logger = logging.getLogger(__name__)

PROVIDER_TIMEOUT_CONNECT = 5.0
BACKOFF_BASE = 1.0
BACKOFF_CAP = 30.0


class TranscriptionService:
    """Primary provider with bounded retries, then the optional fallback.

    A chunk that fails everywhere resolves to zero segments, the line is
    still emitted so line IDs stay gapless (docs/03).
    """

    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        timeout = httpx.Timeout(
            connect=PROVIDER_TIMEOUT_CONNECT,
            read=config.request_timeout_seconds,
            write=config.request_timeout_seconds,
            pool=PROVIDER_TIMEOUT_CONNECT,
        )
        self._client = httpx.AsyncClient(timeout=timeout)
        names = [config.provider] + ([config.fallback_provider] if config.fallback_provider else [])
        self._providers = []
        for name in names:
            cls = get_provider_class(name)
            if cls is None:  # config validation prevents this; belt and braces
                raise ValueError(f"unknown transcription provider {name!r}")
            self._providers.append(cls(config.options_for(name), config.language, self._client))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def transcribe(self, wav: bytes) -> list[RelativeSegment] | None:
        """Returns segments (possibly empty for silence), or None when every
        provider failed and the chunk's audio could not be transcribed."""
        for provider in self._providers:
            try:
                return await self._with_retries(provider, wav)
            except TranscriptionError as exc:
                logger.error("%s gave up on chunk: %s", provider.name, exc)
        return None

    async def _with_retries(self, provider: Any, wav: bytes) -> list[RelativeSegment]:
        attempts = max(1, self._config.max_retries + 1)
        last: TranscriptionError | None = None
        for attempt in range(attempts):
            try:
                return await provider.transcribe(wav)
            except TranscriptionError as exc:
                last = exc
                if not exc.retryable:
                    raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A malformed provider response (unexpected JSON shape, null
                # fields) must behave like any other transient provider
                # failure, retry, fall back, never escape into the pipeline.
                last = TranscriptionError(f"{provider.name}: unexpected error {exc!r}", retryable=True)
            if attempt < attempts - 1:
                delay = min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt)) * (0.5 + random.random())
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    provider.name,
                    attempt + 1,
                    attempts,
                    last,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last is not None
        raise last
