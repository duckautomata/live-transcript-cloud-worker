from __future__ import annotations

import httpx
import pytest

from live_transcript_cloud_worker import transcribe as tr
from live_transcript_cloud_worker.config import TranscriptionConfig
from live_transcript_cloud_worker.providers import (
    TranscriptionError,
    get_provider_class,
    registered_names,
)
from live_transcript_cloud_worker.providers.cloudflare import CloudflareProvider
from live_transcript_cloud_worker.providers.decensor import decensor
from live_transcript_cloud_worker.providers.deepgram import DeepgramProvider, _group_words
from live_transcript_cloud_worker.transcribe import TranscriptionService

CF_OPTIONS = {"account_id": "acct", "api_token": "tok"}
DG_OPTIONS = {"api_key": "key"}


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(tr, "BACKOFF_BASE", 0.0)
    monkeypatch.setattr(tr, "BACKOFF_CAP", 0.0)


def scripted_client(responses: list) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body) if isinstance(body, dict) else httpx.Response(status, text=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests


# ------------------------------------------------------------------ registry


def test_builtin_providers_registered():
    assert registered_names() == ("cloudflare", "deepgram")
    assert get_provider_class("cloudflare") is CloudflareProvider
    assert get_provider_class("deepgram") is DeepgramProvider
    assert get_provider_class("acme") is None


def test_register_rejects_duplicate_name():
    from live_transcript_cloud_worker.providers.base import register

    with pytest.raises(ValueError, match="duplicate provider name"):

        @register
        class Impostor(DeepgramProvider):
            name = "deepgram"

    # The original registration is untouched.
    assert get_provider_class("deepgram") is DeepgramProvider


def test_validate_flags_unknown_options_and_missing_credentials():
    assert CloudflareProvider.validate(CF_OPTIONS) == []
    problems = CloudflareProvider.validate({"account_id": "a", "api_tokn": "typo"})
    assert any("unknown option" in p for p in problems)
    assert any("api_token" in p for p in problems)
    assert DeepgramProvider.validate({}) == ["needs api_key (or DEEPGRAM_API_KEY)"]


# ------------------------------------------------------------------ decensor


def test_decensor_maps_lower_and_capitalised():
    assert decensor("f**k this") == "fuck this"
    assert decensor("Sh** happens") == "Shit happens"
    assert decensor("clean text") == "clean text"


# ---------------------------------------------------------------- cloudflare


async def test_cloudflare_parses_segments():
    client, requests = scripted_client(
        [
            (
                200,
                {
                    "result": {
                        "text": "hello there general",
                        "segments": [
                            {"start": 0.0, "end": 2.0, "text": " hello there "},
                            {"start": 2.5, "end": 4.0, "text": "general"},
                        ],
                    },
                    "success": True,
                },
            ),
        ]
    )
    cf = CloudflareProvider(CF_OPTIONS, "en", client)
    segments = await cf.transcribe(b"wav")
    assert segments == [(0.0, "hello there"), (2.5, "general")]
    assert "acct/ai/run/@cf/openai/whisper-large-v3-turbo" in str(requests[0].url)
    assert requests[0].headers["Authorization"] == "Bearer tok"


async def test_cloudflare_model_option_changes_url():
    client, requests = scripted_client([(200, {"result": {"text": "hi"}, "success": True})])
    cf = CloudflareProvider({**CF_OPTIONS, "model": "@cf/openai/whisper"}, "en", client)
    await cf.transcribe(b"wav")
    assert str(requests[0].url).endswith("/ai/run/@cf/openai/whisper")


async def test_cloudflare_text_only_fallback():
    client, _ = scripted_client([(200, {"result": {"text": "just text"}, "success": True})])
    cf = CloudflareProvider(CF_OPTIONS, "en", client)
    assert await cf.transcribe(b"wav") == [(0.0, "just text")]


async def test_cloudflare_silence_gives_no_segments():
    client, _ = scripted_client([(200, {"result": {"text": ""}, "success": True})])
    cf = CloudflareProvider(CF_OPTIONS, "en", client)
    assert await cf.transcribe(b"wav") == []


async def test_cloudflare_capacity_429_retryable_quota_not():
    client, _ = scripted_client(
        [
            (429, {"result": None, "success": False, "errors": [{"code": 3040, "message": "capacity"}]}),
        ]
    )
    cf = CloudflareProvider(CF_OPTIONS, "en", client)
    with pytest.raises(TranscriptionError) as e:
        await cf.transcribe(b"wav")
    assert e.value.retryable

    client, _ = scripted_client(
        [
            (429, {"result": None, "success": False, "errors": [{"code": 3036, "message": "quota"}]}),
        ]
    )
    cf = CloudflareProvider(CF_OPTIONS, "en", client)
    with pytest.raises(TranscriptionError) as e:
        await cf.transcribe(b"wav")
    assert not e.value.retryable


# ------------------------------------------------------------------ deepgram

DG_OK = {
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "Hello there. General Kenobi.",
                        "confidence": 0.99,
                        "words": [
                            {"word": "hello", "punctuated_word": "Hello", "start": 0.1, "end": 0.4},
                            {"word": "there", "punctuated_word": "there.", "start": 0.45, "end": 0.7},
                            {"word": "general", "punctuated_word": "General", "start": 2.5, "end": 2.9},
                            {"word": "kenobi", "punctuated_word": "Kenobi.", "start": 2.95, "end": 3.4},
                        ],
                    }
                ]
            }
        ]
    },
}


async def test_deepgram_groups_words_on_gaps():
    client, requests = scripted_client([(200, DG_OK)])
    dg = DeepgramProvider(DG_OPTIONS, "en", client)
    segments = await dg.transcribe(b"wav")
    assert segments == [(0.1, "Hello there."), (2.5, "General Kenobi.")]
    request = requests[0]
    assert request.headers["Authorization"] == "Token key"
    assert request.headers["Content-Type"] == "audio/wav"
    params = httpx.QueryParams(request.url.query.decode())
    assert params["model"] == "nova-3"
    assert params["smart_format"] == "true"


async def test_deepgram_empty_transcript_is_valid():
    client, _ = scripted_client(
        [
            (200, {"results": {"channels": [{"alternatives": [{"transcript": "", "words": []}]}]}}),
        ]
    )
    dg = DeepgramProvider(DG_OPTIONS, "en", client)
    assert await dg.transcribe(b"wav") == []


async def test_deepgram_retryability():
    for status, retryable in ((429, True), (500, True), (422, True), (402, False), (401, False)):
        client, _ = scripted_client([(status, {"err_code": "X", "err_msg": "y"})])
        dg = DeepgramProvider(DG_OPTIONS, "en", client)
        with pytest.raises(TranscriptionError) as e:
            await dg.transcribe(b"wav")
        assert e.value.retryable is retryable, f"status {status}"


async def test_deepgram_keyterms_repeated():
    client, requests = scripted_client([(200, DG_OK)])
    dg = DeepgramProvider({**DG_OPTIONS, "keyterms": ["Foo", "Bar Baz"]}, "en", client)
    await dg.transcribe(b"wav")
    query = requests[0].url.query.decode()
    assert query.count("keyterm=") == 2


def test_group_words_single_segment():
    words = [
        {"word": "a", "start": 0.0, "end": 0.2},
        {"word": "b", "start": 0.3, "end": 0.5},
    ]
    assert _group_words(words) == [(0.0, "a b")]


# ------------------------------------------------------------------- service


def sconfig(**kwargs) -> TranscriptionConfig:
    kwargs.setdefault(
        "provider_options",
        {"deepgram": dict(DG_OPTIONS), "cloudflare": dict(CF_OPTIONS)},
    )
    return TranscriptionConfig(**kwargs)


class FlakyProvider:
    name = "flaky"

    def __init__(self, failures: int, retryable: bool = True):
        self.failures = failures
        self.retryable = retryable
        self.calls = 0

    async def transcribe(self, wav: bytes):
        self.calls += 1
        if self.calls <= self.failures:
            raise TranscriptionError("nope", retryable=self.retryable)
        return [(0.0, "recovered")]


def test_service_builds_providers_from_registry():
    service = TranscriptionService(sconfig(provider="deepgram", fallback_provider="cloudflare"))
    assert [p.name for p in service._providers] == ["deepgram", "cloudflare"]


async def test_service_retries_until_success():
    service = TranscriptionService.__new__(TranscriptionService)
    service._config = sconfig(max_retries=4)
    provider = FlakyProvider(failures=2)
    service._providers = [provider]
    assert await service.transcribe(b"wav") == [(0.0, "recovered")]
    assert provider.calls == 3


async def test_service_falls_back_to_secondary():
    service = TranscriptionService.__new__(TranscriptionService)
    service._config = sconfig(max_retries=1)
    primary = FlakyProvider(failures=99)
    secondary = FlakyProvider(failures=0)
    service._providers = [primary, secondary]
    assert await service.transcribe(b"wav") == [(0.0, "recovered")]
    assert primary.calls == 2  # initial + 1 retry
    assert secondary.calls == 1


async def test_service_non_retryable_skips_straight_to_fallback():
    service = TranscriptionService.__new__(TranscriptionService)
    service._config = sconfig(max_retries=5)
    primary = FlakyProvider(failures=99, retryable=False)
    secondary = FlakyProvider(failures=0)
    service._providers = [primary, secondary]
    assert await service.transcribe(b"wav") == [(0.0, "recovered")]
    assert primary.calls == 1


async def test_service_total_failure_returns_none():
    service = TranscriptionService.__new__(TranscriptionService)
    service._config = sconfig(max_retries=0)
    service._providers = [FlakyProvider(failures=99)]
    assert await service.transcribe(b"wav") is None
