"""Deepgram pre-recorded transcription via POST /v1/listen.

Raw WAV body, ``Authorization: Token`` auth. Retry 408/422/429/5xx; 402
(out of credits) is terminal and loud. Silence legitimately returns an
empty transcript, a result, not an error.
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import RelativeSegment, TranscriptionError, TranscriptionProvider, register


@register
class DeepgramProvider(TranscriptionProvider):
    name = "deepgram"
    known_options = frozenset({"api_key", "model", "smart_format", "filler_words", "keyterms"})
    env_overrides = {"api_key": "DEEPGRAM_API_KEY"}

    @classmethod
    def validate_options(cls, options):
        if not options.get("api_key"):
            return ["needs api_key (or DEEPGRAM_API_KEY)"]
        return []

    def __init__(self, options, language, client) -> None:
        super().__init__(options, language, client)
        self._headers = {
            "Authorization": f"Token {options['api_key']}",
            "Content-Type": "audio/wav",
        }
        # Element type matches httpx's params typing exactly, a narrower
        # list[tuple[str, str]] is not assignable (lists are invariant).
        params: list[tuple[str, str | int | float | bool | None]] = [
            ("model", str(options.get("model", "nova-3"))),
            ("language", language),
            ("smart_format", "true" if options.get("smart_format", True) else "false"),
        ]
        if options.get("filler_words", False):
            params.append(("filler_words", "true"))
        for term in options.get("keyterms") or ():
            params.append(("keyterm", str(term)))
        # QueryParams preserves the repeated keyterm keys and satisfies
        # httpx's params typing (a plain list[tuple[str, str]] does not:
        # lists are invariant in their element type).
        self._params = httpx.QueryParams(params)

    async def transcribe(self, wav: bytes) -> list[RelativeSegment]:
        try:
            response = await self.client.post(
                "https://api.deepgram.com/v1/listen",
                params=self._params,
                headers=self._headers,
                content=wav,
            )
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"deepgram: {type(exc).__name__}: {exc}", retryable=True) from exc

        status = response.status_code
        if status != 200:
            message = f"deepgram: HTTP {status} {response.text[:200]}"
            if status == 402:
                raise TranscriptionError(f"{message}, out of credits", retryable=False)
            if status in (408, 422, 429) or status >= 500:
                raise TranscriptionError(message, retryable=True)
            raise TranscriptionError(message, retryable=False)

        try:
            alternatives = response.json()["results"]["channels"][0]["alternatives"]
            alt = alternatives[0] if alternatives else {}
        except (ValueError, KeyError, IndexError, TypeError):
            raise TranscriptionError("deepgram: unexpected response shape", retryable=True) from None

        words = alt.get("words") or []
        if words:
            return _group_words(words)
        transcript = str(alt.get("transcript", "")).strip()
        return [(0.0, transcript)] if transcript else []


def _group_words(words: list[dict[str, Any]], gap: float = 1.0) -> list[RelativeSegment]:
    """Group word timings into segments, splitting at silences >= ``gap``."""
    segments: list[RelativeSegment] = []
    current: list[str] = []
    seg_start = 0.0
    prev_end = 0.0
    for w in words:
        text = str(w.get("punctuated_word") or w.get("word") or "").strip()
        if not text:
            continue
        start = float(w.get("start", prev_end))
        if not current:
            seg_start = start
        elif start - prev_end >= gap:
            segments.append((seg_start, " ".join(current)))
            current = []
            seg_start = start
        current.append(text)
        prev_end = float(w.get("end", start))
    if current:
        segments.append((seg_start, " ".join(current)))
    return segments
