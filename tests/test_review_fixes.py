"""Regression tests for the confirmed review findings."""

from __future__ import annotations

import httpx
import pytest
from conftest import make_config
from test_pipeline import FakeServer, build, make_chunk, run_pipeline
from test_server_client import Script

from live_transcript_cloud_worker import server_client as sc
from live_transcript_cloud_worker import transcribe as tr
from live_transcript_cloud_worker.config import TranscriptionConfig
from live_transcript_cloud_worker.server_client import (
    LineResult,
    MediaResult,
    ServerClient,
)
from live_transcript_cloud_worker.transcribe import TranscriptionService
from live_transcript_cloud_worker.uploader import MediaUploader


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(sc, "_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(tr, "BACKOFF_BASE", 0.0)
    monkeypatch.setattr(tr, "BACKOFF_CAP", 0.0)


# --- finding: media must not dispatch when the line never landed ------------


async def test_failed_line_parks_media_without_dispatch(tmp_path):
    server = FakeServer()
    server.line_results = [LineResult.FAILED, LineResult.OK]
    pipeline, state, server, uploader = build(tmp_path, server=server)
    chunks = [make_chunk(tmp_path, i) for i in range(2)]
    await run_pipeline(pipeline, chunks)
    # Only the delivered line's media is dispatched...
    assert [(sid, lid) for _, sid, lid, _ in uploader.enqueued] == [("abc", 1)]
    # ...but the undelivered line's file is parked durably, not deleted.
    parked = state.pending_media()
    assert [(p.stream_id, p.line_id) for p in parked] == [("abc", 0), ("abc", 1)]


async def test_conflict_with_failed_sync_parks_media(tmp_path):
    server = FakeServer()
    server.line_results = [LineResult.CONFLICT]
    server.sync_ok = False
    pipeline, state, _, uploader = build(tmp_path, server=server)
    await run_pipeline(pipeline, [make_chunk(tmp_path, 0)])
    assert uploader.enqueued == []
    assert len(state.pending_media()) == 1


# --- finding: network-error media exhaustion keeps the file -----------------


async def test_media_network_exhaustion_returns_retry_later(tmp_path):
    media = tmp_path / "m.ts"
    media.write_bytes(b"bytes")
    script = Script([httpx.ConnectError("down")] * 4)
    config = make_config(tmp_path)
    client = ServerClient(config, transport=httpx.MockTransport(script.handler))
    assert await client.upload_media("chan", "abc", 3, media) is MediaResult.RETRY_LATER
    assert media.exists()


async def test_media_500_exhaustion_still_drops(tmp_path):
    media = tmp_path / "m.ts"
    media.write_bytes(b"bytes")
    script = Script([(500, "ffmpeg")] * 4)
    config = make_config(tmp_path)
    client = ServerClient(config, transport=httpx.MockTransport(script.handler))
    assert await client.upload_media("chan", "abc", 3, media) is MediaResult.DROP


# --- finding: uploader requeues RETRY_LATER and keeps the file --------------


class OutageServer:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    async def upload_media(self, key, stream_id, line_id, path):
        self.calls += 1
        if self.calls <= self.failures:
            return MediaResult.RETRY_LATER
        return MediaResult.OK


async def test_uploader_retry_later_requeues_until_server_returns(tmp_path, monkeypatch):
    import live_transcript_cloud_worker.uploader as up

    monkeypatch.setattr(up, "RETRY_BACKOFF_BASE", 0.01)
    monkeypatch.setattr(up, "RETRY_BACKOFF_CAP", 0.01)
    monkeypatch.setattr(up, "IDLE_RECHECK_SECONDS", 0.02)
    config = make_config(tmp_path)
    server = OutageServer(failures=2)
    uploader = MediaUploader(config, server)
    uploader.start()
    path = tmp_path / "m.ts"
    path.write_bytes(b"x")
    uploader.enqueue("a", "s", 0, path)
    assert await uploader.drain(5)
    await uploader.stop()
    assert server.calls == 3
    assert not path.exists()  # uploaded on the third try, then removed


# --- finding: pipeline survives per-chunk internal errors -------------------


async def test_pipeline_survives_broken_transcription_result(tmp_path):
    class BrokenThenGood:
        calls = 0

        async def transcribe(self, wav):
            BrokenThenGood.calls += 1
            if BrokenThenGood.calls == 1:
                return [("not-a-float", "boom")]  # breaks Segment building
            return [(0.0, "fine")]

    pipeline, state, server, _ = build(tmp_path, transcriber=BrokenThenGood())
    chunks = [make_chunk(tmp_path, i) for i in range(2)]
    await run_pipeline(pipeline, chunks)
    # The broken chunk is dropped before an ID is assigned; the next chunk
    # takes line 0 and the sequence stays gapless.
    assert [wire["id"] for _, wire in server.lines] == [0]
    assert server.lines[0][1]["segments"][0]["text"] == "fine"
    assert not chunks[0].path.exists()


# --- finding: unexpected provider exceptions retry and fall back ------------


class ExplodingProvider:
    name = "exploding"
    calls = 0

    async def transcribe(self, wav):
        ExplodingProvider.calls += 1
        raise KeyError("results")  # not a TranscriptionError


class GoodProvider:
    name = "good"

    async def transcribe(self, wav):
        return [(0.0, "ok")]


async def test_unexpected_provider_exception_is_retried_then_falls_back():
    service = TranscriptionService.__new__(TranscriptionService)
    service._config = TranscriptionConfig(max_retries=1, provider_options={"deepgram": {"api_key": "k"}})
    ExplodingProvider.calls = 0
    service._providers = [ExplodingProvider(), GoodProvider()]
    assert await service.transcribe(b"wav") == [(0.0, "ok")]
    assert ExplodingProvider.calls == 2  # initial + 1 retry, then fallback


# --- finding: /events tolerates non-dict 200 bodies -------------------------


async def test_events_non_dict_body_returns_none(tmp_path):
    script = Script([(200, ["not", "a", "dict"])])
    config = make_config(tmp_path)
    client = ServerClient(config, transport=httpx.MockTransport(script.handler))
    assert await client.get_events(["chan"], 0, 25) is None
