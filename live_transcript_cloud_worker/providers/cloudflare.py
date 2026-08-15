"""Cloudflare Workers AI whisper via the REST API.

Model defaults to ``@cf/openai/whisper-large-v3-turbo``: JSON body with
base64 audio, Bearer auth. 429 responses carry an error code that matters,
3040 ("capacity temporarily exceeded") is transient and retried, 3036 (daily
neuron quota exhausted) is not: retrying burns time until 00:00 UTC.
"""

from __future__ import annotations

import base64

import httpx

from .base import RelativeSegment, TranscriptionError, TranscriptionProvider, register
from .decensor import decensor


@register
class CloudflareProvider(TranscriptionProvider):
    name = "cloudflare"
    known_options = frozenset({"account_id", "api_token", "model", "vad_filter"})
    env_overrides = {
        "account_id": "CLOUDFLARE_ACCOUNT_ID",
        "api_token": "CLOUDFLARE_API_TOKEN",
    }

    @classmethod
    def validate_options(cls, options):
        if not options.get("account_id") or not options.get("api_token"):
            return ["needs account_id and api_token (or CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN)"]
        return []

    def __init__(self, options, language, client) -> None:
        super().__init__(options, language, client)
        model = options.get("model", "@cf/openai/whisper-large-v3-turbo")
        self._url = f"https://api.cloudflare.com/client/v4/accounts/{options['account_id']}/ai/run/{model}"
        self._headers = {"Authorization": f"Bearer {options['api_token']}"}
        self._vad_filter = bool(options.get("vad_filter", True))

    async def transcribe(self, wav: bytes) -> list[RelativeSegment]:
        body = {
            "audio": base64.b64encode(wav).decode("ascii"),
            "task": "transcribe",
            "language": self.language,
            # Mitigations for whisper's hallucination-on-silence failure mode.
            "vad_filter": self._vad_filter,
            "condition_on_previous_text": False,
        }
        try:
            response = await self.client.post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"cloudflare: {type(exc).__name__}: {exc}", retryable=True) from exc

        if response.status_code != 200:
            raise self._classify_error(response)

        try:
            result = response.json().get("result") or {}
        except ValueError:
            raise TranscriptionError("cloudflare: unparseable response body", retryable=True) from None

        segments = result.get("segments")
        if isinstance(segments, list) and segments:
            out = []
            for seg in segments:
                text = decensor(str(seg.get("text", "")).strip())
                if text:
                    out.append((float(seg.get("start", 0.0)), text))
            return out
        text = decensor(str(result.get("text", "")).strip())
        return [(0.0, text)] if text else []

    def _classify_error(self, response: httpx.Response) -> TranscriptionError:
        status = response.status_code
        code = None
        try:
            errors = response.json().get("errors") or []
            if errors:
                code = errors[0].get("code")
        except (ValueError, AttributeError, IndexError, TypeError):
            pass
        message = f"cloudflare: HTTP {status} code={code} {response.text[:200]}"
        if status == 429:
            # 3040 = transient capacity; 3036 = daily quota exhausted.
            return TranscriptionError(message, retryable=code != 3036)
        if status == 408 or status >= 500:
            return TranscriptionError(message, retryable=True)
        return TranscriptionError(message, retryable=False)
