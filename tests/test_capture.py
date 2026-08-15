"""Capture tests: real ffmpeg segmenting driven by a scripted fake yt-dlp."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from conftest import make_config

from live_transcript_cloud_worker import capture
from live_transcript_cloud_worker.config import (
    CaptureConfig,
    ServerConfig,
    StaleThresholdConfig,
    StreamerConfig,
)
from live_transcript_cloud_worker.models import MediaType, StreamInfo
from live_transcript_cloud_worker.state import ChannelState

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")

INFO = StreamInfo(url="https://example.test/live", stream_id="stream1", title="T", is_live=True)


@pytest.fixture(scope="module")
def tone_ts(tmp_path_factory) -> Path:
    """~14s AAC tone in MPEG-TS, the shape a yt-dlp download produces."""
    path = tmp_path_factory.mktemp("fixtures") / "tone.ts"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=14",
            "-c:a",
            "aac",
            "-f",
            "mpegts",
            str(path),
        ],
        check=True,
    )
    return path


def fake_ytdlp(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake-yt-dlp"
    script.write_text(f"#!/bin/bash\n{body}\n")
    script.chmod(0o755)
    return script


def capture_config(tmp_path: Path, ytdlp: Path, ytdlp_stale: float = 180.0):
    return make_config(
        tmp_path,
        server=ServerConfig(
            api_key="k",
            url="http://s.test",
            enabled=True,
            buffer_size_seconds=4.0,
            stale_threshold=StaleThresholdConfig(ytdlp_seconds=ytdlp_stale),
        ),
        capture=CaptureConfig(yt_dlp_path=str(ytdlp), ffmpeg_path=ffmpeg, live_latency_seconds=1.0),
    )


STREAMER = StreamerConfig(key="chan", urls=("u",), media_type=MediaType.AUDIO)


async def run_capture(config, state, should_stop=lambda: False):
    emitted = []

    async def emit(chunk):
        emitted.append(chunk)

    stats = await asyncio.wait_for(
        capture.capture_stream(config, STREAMER, state, INFO, emit, should_stop),
        timeout=60,
    )
    return stats, emitted


async def test_capture_segments_whole_stream(tmp_path, tone_ts):
    ytdlp = fake_ytdlp(tmp_path, f'exec cat "{tone_ts}"')
    config = capture_config(tmp_path, ytdlp)
    state = ChannelState("chan", tmp_path / "tmp" / "chan")

    stats, emitted = await run_capture(config, state)

    # ~14s at 4s segments -> 3 full chunks + a tail (>= min_chunk 0.5s).
    assert 3 <= len(emitted) <= 5
    assert stats.segments_emitted == len(emitted)
    total = sum(c.duration for c in emitted)
    assert 12.5 <= total <= 15.0
    for chunk in emitted:
        assert chunk.stream_id == "stream1"
        assert chunk.vod_accurate is False
        assert chunk.media_type is MediaType.AUDIO
        assert chunk.path.exists()  # emitted files stay for the pipeline
        assert len(chunk.pcm) > 0
        # mtime anchor: start = mtime - duration - live_latency(1.0)
        mtime = chunk.path.stat().st_mtime
        assert chunk.audio_start_time == pytest.approx(mtime - chunk.duration - 1.0, abs=0.01)
    # Chunks arrive in stream order.
    names = [c.path.name for c in emitted]
    assert names == sorted(names)


async def test_capture_watchdog_terminates_wedged_ytdlp(tmp_path):
    ytdlp = fake_ytdlp(tmp_path, "exec sleep 300")  # alive, produces nothing
    config = capture_config(tmp_path, ytdlp, ytdlp_stale=1.0)
    state = ChannelState("chan", tmp_path / "tmp" / "chan")

    started = time.monotonic()
    stats, emitted = await run_capture(config, state)
    assert emitted == []
    assert time.monotonic() - started < 20  # watchdog fired, no 300s hang


async def test_capture_stop_signal_flushes_and_returns(tmp_path, tone_ts):
    # Stream the tone then hold the pipe open forever, like a live stream.
    ytdlp = fake_ytdlp(tmp_path, f'cat "{tone_ts}"\nexec sleep 300')
    config = capture_config(tmp_path, ytdlp)
    state = ChannelState("chan", tmp_path / "tmp" / "chan")

    stop = {"flag": False}
    emitted = []

    async def emit(chunk):
        emitted.append(chunk)
        stop["flag"] = True  # request stop after the first chunk

    await asyncio.wait_for(
        capture.capture_stream(config, STREAMER, state, INFO, emit, lambda: stop["flag"]),
        timeout=60,
    )
    # Terminating yt-dlp closes the pipe; ffmpeg flushes the remaining
    # buffered segments and they are all consumed before returning.
    assert len(emitted) >= 1


async def test_capture_spawn_failure_is_contained(tmp_path):
    config = capture_config(tmp_path, tmp_path / "does-not-exist")
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    stats, emitted = await run_capture(config, state)
    assert stats.segments_emitted == 0
    assert emitted == []


async def test_consume_segment_garbage_skipped(tmp_path):
    config = capture_config(tmp_path, tmp_path / "unused")
    garbage = tmp_path / "chunk000000.ts"
    garbage.write_bytes(b"\x00" * 2048)
    chunk = await capture._consume_segment(config, STREAMER, INFO, garbage)
    assert chunk is None
    assert not garbage.exists()
