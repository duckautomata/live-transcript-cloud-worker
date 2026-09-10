"""Contract tests for retry classification (docs/06) via httpx.MockTransport."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from conftest import make_config

from live_transcript_cloud_worker import server_client as sc
from live_transcript_cloud_worker.models import Line, MediaType, Segment
from live_transcript_cloud_worker.server_client import (
    LineResult,
    MediaResult,
    ServerClient,
)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(sc, "_backoff", lambda *a, **k: 0.0)


class Script:
    """Scripted responses; records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected extra request: {request.method} {request.url}")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        if isinstance(body, (dict, list)):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body or "")


def client_with(tmp_path: Path, *responses) -> tuple[ServerClient, Script]:
    script = Script(responses)
    config = make_config(tmp_path)
    return ServerClient(config, transport=httpx.MockTransport(script.handler)), script


LINE = Line(id=0, timestamp=100, segments=(Segment(100, "hi"),), vod_accurate=True)


async def test_activate_sends_expected_params(tmp_path):
    client, script = client_with(tmp_path, (200, "ok"))
    assert await client.activate("chan", "abc", "Title", 1000, MediaType.AUDIO)
    request = script.requests[0]
    assert request.url.path == "/chan/activate"
    params = dict(httpx.QueryParams(request.url.query.decode()))
    assert params == {"id": "abc", "title": "Title", "startTime": "1000", "mediaType": "audio"}
    assert request.headers["X-API-Key"] == "test-key"


async def test_208_is_success(tmp_path):
    client, _ = client_with(tmp_path, (208, "already"))
    assert await client.activate("chan", "abc", "T", 1000, MediaType.NONE)


async def test_5xx_retried_then_success(tmp_path):
    client, script = client_with(tmp_path, (500, "boom"), (502, "bad"), (200, "ok"))
    assert await client.activate("chan", "abc", "T", 1000, MediaType.NONE)
    assert len(script.requests) == 3


async def test_400_not_retried(tmp_path):
    client, script = client_with(tmp_path, (400, "bad"))
    assert not await client.activate("chan", "abc", "T", 1000, MediaType.NONE)
    assert len(script.requests) == 1


async def test_403_not_retried(tmp_path):
    client, script = client_with(tmp_path, (403, "Forbidden"))
    assert not await client.activate("chan", "abc", "T", 1000, MediaType.NONE)
    assert len(script.requests) == 1


async def test_network_error_retried(tmp_path):
    client, script = client_with(tmp_path, httpx.ConnectError("refused"), (200, "ok"))
    assert await client.activate("chan", "abc", "T", 1000, MediaType.NONE)
    assert len(script.requests) == 2


async def test_line_ok(tmp_path):
    client, script = client_with(tmp_path, (200, "ok"))
    assert await client.post_line("chan", "abc", LINE) is LineResult.OK
    body = json.loads(script.requests[0].content)
    assert body["id"] == 0
    assert body["mediaAvailable"] is False
    assert body["segments"] == [{"timestamp": 100, "text": "hi"}]


async def test_line_409_is_conflict_not_retried(tmp_path):
    client, script = client_with(tmp_path, (409, "conflict"))
    assert await client.post_line("chan", "abc", LINE) is LineResult.CONFLICT
    assert len(script.requests) == 1


async def test_line_exhausted_retries_fails(tmp_path):
    client, script = client_with(tmp_path, *([(500, "boom")] * 5))
    assert await client.post_line("chan", "abc", LINE) is LineResult.FAILED
    assert len(script.requests) == 5


async def test_media_404_is_retried(tmp_path):
    media = tmp_path / "m.ts"
    media.write_bytes(b"bytes")
    client, script = client_with(tmp_path, (404, "Stream not found"), (200, "ok"))
    assert await client.upload_media("chan", "abc", 3, media) is MediaResult.OK
    assert len(script.requests) == 2
    assert script.requests[0].url.path == "/chan/media/abc/3"


async def test_media_400_drops(tmp_path):
    media = tmp_path / "m.ts"
    media.write_bytes(b"bytes")
    client, script = client_with(tmp_path, (400, "bad"))
    assert await client.upload_media("chan", "abc", 3, media) is MediaResult.DROP
    assert len(script.requests) == 1


async def test_media_500_bounded_then_drop(tmp_path):
    media = tmp_path / "m.ts"
    media.write_bytes(b"bytes")
    client, script = client_with(tmp_path, *([(500, "ffmpeg")] * 4))
    assert await client.upload_media("chan", "abc", 3, media) is MediaResult.DROP
    assert len(script.requests) == 4


async def test_events_cursor_roundtrip(tmp_path):
    client, script = client_with(tmp_path, (200, {"cursor": 1234, "events": {"chan": ["incoming", "restart"]}}))
    events, cursor = await client.get_events(["chan"], 5, 25)
    assert cursor == 1234
    assert events == {"chan": ["incoming", "restart"]}
    params = dict(httpx.QueryParams(script.requests[0].url.query.decode()))
    assert params == {"channels": "chan", "since": "5", "wait": "25"}


async def test_events_204_keeps_cursor(tmp_path):
    client, _ = client_with(tmp_path, (204, ""))
    events, cursor = await client.get_events(["chan"], 77, 25)
    assert events == {}
    assert cursor == 77


async def test_events_failure_returns_none(tmp_path):
    client, _ = client_with(tmp_path, (500, "db"))
    assert await client.get_events(["chan"], 0, 25) is None


async def test_delete_incoming_404_is_success(tmp_path):
    client, _ = client_with(tmp_path, (404, "gone"))
    assert await client.delete_incoming("chan", "https://u")


async def test_ack_restart_404_is_success(tmp_path):
    client, _ = client_with(tmp_path, (404, "none"))
    assert await client.ack_restart("chan")


async def test_deactivate_empty_id_skips_request(tmp_path):
    client, script = client_with(tmp_path)
    assert await client.deactivate("chan", "")
    assert script.requests == []


async def test_sync_payload_posted(tmp_path):
    client, script = client_with(tmp_path, (200, "ok"))
    payload = {"streamId": "abc", "transcript": []}
    assert await client.sync("chan", payload)
    assert json.loads(script.requests[0].content) == payload


async def test_429_respects_retry_after(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(sc, "_backoff", lambda a, ra=None: delays.append(ra) or 0.0)
    client, script = client_with(tmp_path, (429, "slow down"), (200, "ok"))
    assert await client.activate("chan", "abc", "T", 1000, MediaType.NONE)
    assert len(script.requests) == 2


# ----------------------------- transport failure vs an authoritative answer


async def test_get_incoming_returns_none_when_unreachable(tmp_path):
    # None, not []: the watcher drops every URL it knows about when the queue
    # reads empty, so a transport failure must stay distinguishable from one.
    client, script = client_with(
        tmp_path,
        httpx.ConnectError("All connection attempts failed"),
        httpx.ConnectError("All connection attempts failed"),
    )
    assert await client.get_incoming("chan") is None
    assert len(script.requests) == 2


async def test_get_incoming_returns_empty_list_when_server_says_empty(tmp_path):
    client, _ = client_with(tmp_path, (200, {"urls": []}))
    assert await client.get_incoming("chan") == []


async def test_get_incoming_unparseable_body_is_not_an_empty_queue(tmp_path):
    client, _ = client_with(tmp_path, (200, "not json"))
    assert await client.get_incoming("chan") is None


async def test_get_restart_retries_once(tmp_path):
    # get_restart only ever runs after /events already failed, so a single
    # attempt is guaranteed to be the worst-timed one.
    client, script = client_with(
        tmp_path,
        httpx.ConnectError("refused"),
        (200, {"pending": True}),
    )
    assert await client.get_restart("chan") is True
    assert len(script.requests) == 2


async def test_no_backoff_sleep_after_the_final_attempt(tmp_path, monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(sc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sc, "_backoff", lambda *a, **k: 1.0)

    client, script = client_with(
        tmp_path,
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
    )
    assert await client.get_incoming("chan") is None
    # Two attempts, one gap between them: the trailing sleep bought nothing
    # and only widened every observed outage.
    assert len(script.requests) == 2
    assert slept == [1.0]


async def test_final_retry_after_is_still_honoured(tmp_path, monkeypatch):
    # Skipping the last sleep must not drop server-requested backpressure.
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(sc.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sc, "_backoff", lambda attempt, retry_after=None: 9.0 if retry_after else 1.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "9"}, text="slow down")

    config = make_config(tmp_path)
    client = ServerClient(config, transport=httpx.MockTransport(handler))

    assert await client.get_incoming("chan") is None
    assert slept == [9.0, 9.0]


def test_error_chain_surfaces_the_underlying_errno():
    # "All connection attempts failed" names no errno; the errno is the whole
    # diagnosis, and it only survives on the cause chain.
    cause = ConnectionRefusedError(111, "Connection refused")
    exc = httpx.ConnectError("All connection attempts failed")
    exc.__cause__ = OSError("All connection attempts failed")
    exc.__cause__.__cause__ = cause

    chain = sc._error_chain(exc)
    assert "ConnectionRefusedError" in chain
    assert "111" in chain
