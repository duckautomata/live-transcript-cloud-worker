"""YouTube Timeline capture (capture_timeline): the run loop against a scripted
SegmentSource on a fake clock, with mux/decode faked, plus one end-to-end run
through the real HlsSegmentSource (a fake YouTube over httpx.MockTransport)
and the real ffmpeg."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from conftest import make_config
from test_youtube_timeline_source import FakeYouTube
from test_youtube_timeline_source import _clock as yt_clock
from timeline_fixtures import id3_timestamp_tag, ts_pes_packet, video_segment

from live_transcript_cloud_worker import capture_timeline as ct
from live_transcript_cloud_worker.audio import AudioExtractError
from live_transcript_cloud_worker.capture_timeline import TimelineCapture, TimelineStats
from live_transcript_cloud_worker.config import CaptureConfig, ServerConfig, StaleThresholdConfig, StreamerConfig
from live_transcript_cloud_worker.models import Chunk, MediaType, StreamInfo
from live_transcript_cloud_worker.state import ChannelState
from live_transcript_cloud_worker.youtube_timeline import (
    CONTAINER_ADTS,
    CONTAINER_MPEGTS,
    TRACK_AUDIO,
    TRACK_MUXED,
    TRACK_VIDEO,
    TS_WRAP_SECONDS,
    MuxError,
    SegmentGroup,
    TimelineChunk,
    TrackSegment,
    scan_ts_pts,
)
from live_transcript_cloud_worker.youtube_timeline_source import (
    HLS_CLOCK_TO_VOD_SECONDS,
    HlsSegmentSource,
    ResolvedStream,
    ResolvedTrack,
    ResolveError,
    ResolveStopped,
    SegmentSource,
    SourceError,
    SourceExpiredError,
    SourceResetError,
    StreamNotLiveError,
    StreamUnavailableError,
    UnsupportedSourceError,
)

ffmpeg = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")

START = 1_791_932_469
SEG = 1024 * 43 / 44100  # one 1 s HLS audio segment
WATCH_URL = "https://www.youtube.com/watch?v=abcdefghijk"


class Clock:
    """Monotonic and wall clock moving together. `on_second` runs once for
    every whole second of fake time that passes."""

    def __init__(self, wall_at_zero: float):
        self.t = 0.0
        self.wall_at_zero = wall_at_zero
        self.on_second: Callable[[], None] | None = None
        self._next_second = 1.0

    def mono(self) -> float:
        return self.t

    def wall(self) -> float:
        return self.wall_at_zero + self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)
        while self.on_second is not None and self.t >= self._next_second:
            self._next_second += 1.0
            self.on_second()

    async def asleep(self, seconds: float) -> None:
        self.sleep(seconds)
        await asyncio.sleep(0)  # a real sleep suspends: background tasks (a URL refresh) get to run


def group(sequence, clock, video=True, audio=True):
    segments = {}
    if audio:
        segments[TRACK_AUDIO] = TrackSegment(TRACK_AUDIO, sequence, b"a", CONTAINER_ADTS, clock, SEG, 512 / 44100)
    if video:
        segments[TRACK_VIDEO] = TrackSegment(TRACK_VIDEO, sequence, b"v", CONTAINER_MPEGTS, clock + 0.01, 1.0)
    return SegmentGroup(sequence, segments)


def run_of(first, count, clock0):
    """`count` contiguous groups starting at sequence `first` and clock `clock0`."""
    return [group(first + i, clock0 + i * SEG) for i in range(count)]


def resolved(expires_at=None, tracks=(TRACK_VIDEO, TRACK_AUDIO)):
    return ResolvedStream("abcdefghijk", "is_live", tuple(ResolvedTrack(t, "m3u8_native", f"https://m/{t}", t) for t in tracks), expires_at)


class FakeSource(SegmentSource):
    """Plays a script: a SegmentGroup, None (nothing new, takes `timeout`),
    ("sleep", s) (time passes), "end" (ENDLIST), or an exception to raise."""

    wrap_period = TS_WRAP_SECONDS
    vod_offset = 0.036

    def __init__(self, resolved_stream, resume, ctx, script, clock, step=1.0):
        super().__init__(resolved_stream, resume, ctx)
        self.script = list(script)
        self.clock = clock
        self.step = step
        self.resume = resume
        self.closed = False
        # When set, the live edge's sequence now: behind_live counts 1 s per sequence up to it.
        self.live_head: Callable[[], int] | None = None

    @property
    def behind_live(self):
        if self.live_head is None or self._next_sequence is None:
            return None
        return max(0.0, float(self.live_head() - self._next_sequence + 1))

    async def next_group(self, timeout):
        await asyncio.sleep(0)  # a real source suspends while it waits on the network
        if not self.script:
            self.clock.sleep(timeout)
            return None
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        if action == "end":
            self._ended = True
            return None
        if isinstance(action, tuple):
            self.clock.sleep(action[1])
            return None
        if action is None:
            self.clock.sleep(timeout)
            return None
        self.clock.sleep(self.step)
        self._next_sequence = action.sequence + 1
        return action

    async def aclose(self):
        self.closed = True


@dataclass
class Item:
    """An emitted chunk, with what its file held and when it was emitted."""

    raw: bytes
    chunk: Chunk
    at: float  # fake wall clock

    def __getattr__(self, name):
        if name == "chunk":  # not set yet (copy, pickle): don't recurse
            raise AttributeError(name)
        return getattr(self.chunk, name)


class Harness:
    def __init__(
        self,
        monkeypatch,
        tmp_path,
        start_time: int | None = START,
        stream_id="abcdefghijk",
        media_type=MediaType.VIDEO,
        lfs_gap=600.0,
        fake_media=True,
    ):
        self.config = make_config(
            tmp_path,
            server=ServerConfig(
                api_key="k",
                url="http://s.test",
                buffer_size_seconds=6,
                stale_threshold=StaleThresholdConfig(lfs_gap_seconds=lfs_gap, no_audio_seconds=60),
            ),
            capture=CaptureConfig(ffmpeg_path=ffmpeg or "ffmpeg", live_latency_seconds=1),
        )
        self.streamer = StreamerConfig(
            key="key", urls=("https://www.youtube.com/@chan/live",), media_type=media_type, use_youtube_timeline=True
        )
        self.state = ChannelState("key", tmp_path / "tmp" / "key")
        self.capture = TimelineCapture(self.config, self.streamer, self.state)
        # Wall time a few seconds after the clock values used by default (stream ~5000 s in).
        self.clock = Clock(START + 5000.0)
        cap = self.capture
        cap._monotonic = self.clock.mono  # type: ignore[method-assign]
        cap._wallclock = self.clock.wall  # type: ignore[method-assign]
        cap._sleep = self.clock.asleep  # type: ignore[method-assign]
        self.info = StreamInfo(url="https://www.youtube.com/@chan/live", stream_id=stream_id, is_live=True, start_time=start_time)
        self.resolutions: list = []
        self.resolve_calls: list[float] = []
        self.ytdlp_stderr = ""
        self.scripts: list = []
        self.sources: list[FakeSource] = []
        self.steps: list[float] = []
        self.live_head: Callable[[], int] | None = None  # given to every FakeSource
        self.items: list[Item] = []
        self.mux_calls: list[tuple[str, Path, Path]] = []
        self.pcm_seconds: float | list[float] = 6.0
        self.keep_files = False
        self.stop_after: float | None = None
        self.stats: TimelineStats | None = None
        monkeypatch.setattr(ct, "resolve_stream", self._resolve)
        monkeypatch.setattr(ct, "create_segment_source", self._create)
        if fake_media:
            monkeypatch.setattr(ct, "mux_chunk", self._mux)
            monkeypatch.setattr(ct.audio, "extract_pcm", self._extract)

    async def _resolve(self, config, url, media_type, should_stop, raw_log=None, timeout=90.0):
        self.resolve_calls.append(self.clock.t)
        assert url == WATCH_URL
        assert config is self.config and media_type is self.streamer.media_type
        if self.ytdlp_stderr and raw_log is not None:
            raw_log(self.ytdlp_stderr)
        if not self.resolutions:
            raise StreamNotLiveError("live_status is was_live")
        item = self.resolutions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _create(self, resolved_stream, resume, ctx):
        script = self.scripts.pop(0) if self.scripts else ["end"]
        step = self.steps.pop(0) if self.steps else 1.0
        source = FakeSource(resolved_stream, resume, ctx, script, self.clock, step)
        source.live_head = self.live_head
        self.sources.append(source)
        return source

    async def _mux(self, ffmpeg_path, chunk, workdir, output, timeout=30.0):
        self.mux_calls.append((ffmpeg_path, workdir, output))
        output.write_bytes(b"TS:%d-%d" % (chunk.first_sequence, chunk.last_sequence))

    async def _extract(self, ffmpeg_path, path, timeout=30.0):
        assert path.is_file()
        seconds = self.pcm_seconds.pop(0) if isinstance(self.pcm_seconds, list) else self.pcm_seconds
        return b"\x00\x00" * round(16000 * seconds)

    async def emit(self, chunk: Chunk) -> None:
        self.items.append(Item(chunk.path.read_bytes(), chunk, self.clock.wall()))
        if not self.keep_files:
            chunk.path.unlink()  # what the pipeline does once the line is out

    def should_stop(self) -> bool:
        return self.stop_after is not None and self.clock.t > self.stop_after

    async def run(self) -> list[Item]:
        self.stats = await self.capture.run(self.info, self.emit, self.should_stop)
        return self.items

    @property
    def run_log(self) -> str:
        return (self.state.root / "youtube_timeline.log").read_text()


@pytest.fixture
def h(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


# ----------------------------------------------------------------------
# Timestamps
# ----------------------------------------------------------------------


async def test_lines_are_stamped_from_the_stream_clock(h):
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 13, 4990.0) + ["end"]]
    items = await h.run()
    # 6 + 6 sequences, then the 1-sequence leftover flushed at the end.
    assert [i.raw for i in items] == [b"TS:100-105", b"TS:106-111", b"TS:112-112"]
    for item, first_clock in zip(items, (4990.0, 4990.0 + 6 * SEG, 4990.0 + 12 * SEG), strict=True):
        assert item.vod_accurate is True
        assert item.audio_start_time == pytest.approx(START + first_clock + 0.036)
        assert item.media_type is MediaType.VIDEO and item.key == "key"
        # A clock stamp: the pipeline measures the line's lag from capture, not from the stamp.
        assert item.lag_from is not None and abs(item.lag_from - time.time()) < 60


async def test_resume_point_is_saved_after_each_chunk(h):
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 7, 4990.0) + ["end"]]
    await h.run()
    state = json.loads((h.state.root / "youtube_timeline_state.json").read_text())
    assert state == {"stream_id": "abcdefghijk", "next_sequence": 107, "anchor": [pytest.approx(4990.0 + 7 * SEG), h.clock.wall()]}


async def test_a_new_run_of_the_same_stream_resumes_where_the_last_one_stopped(h, monkeypatch, tmp_path):
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    await h.run()
    h2 = Harness(monkeypatch, tmp_path)
    h2.resolutions = [resolved()]
    await h2.run()
    assert h2.sources[0].resume == 106
    # A different stream on the same key doesn't pick up that resume point.
    h3 = Harness(monkeypatch, tmp_path, stream_id="zzzzzzzzzzz")
    monkeypatch.setattr(ct, "watch_url", lambda url, stream_id: WATCH_URL)
    h3.resolutions = [resolved()]
    await h3.run()
    assert len(h3.sources) == 1 and h3.sources[0].resume is None


@pytest.mark.parametrize("start_time", [None, 0])
async def test_unknown_start_time_is_estimated_but_not_vod_accurate(monkeypatch, tmp_path, caplog, start_time):
    """Deliberate deviation from the reference worker: the watcher activates
    such a stream with the activation time as its startTime, which isn't the
    clock's zero, so the server would place these lines wrongly in the VOD.
    They follow the stream clock, but are never marked VOD-accurate."""
    h = Harness(monkeypatch, tmp_path, start_time=start_time)
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 12, 4990.0) + ["end"]]
    caplog.set_level(logging.WARNING)
    items = await h.run()
    assert [i.vod_accurate for i in items] == [False, False]
    assert h.stats.chunks_not_vod_accurate == 2
    # Still stamped from the clock, which skips outages: lag measured from capture.
    assert all(i.lag_from is not None for i in items)
    # Estimated from arrival: the line lands near the wall clock...
    assert abs(items[0].audio_start_time - (items[0].at - 6)) < 10
    # ...and the next one exactly one chunk of stream clock later.
    assert items[1].audio_start_time - items[0].audio_start_time == pytest.approx(6 * SEG)
    assert caplog.text.count("not marked VOD-accurate") == 1
    assert "estimated the stream start" in caplog.text
    # The clock was never checked against a real start, so no later run may continue it on trust.
    state = json.loads((h.state.root / "youtube_timeline_state.json").read_text())
    assert state == {"stream_id": "abcdefghijk", "next_sequence": 112}


async def test_wrap_count_is_seeded_from_the_stream_start(monkeypatch, tmp_path):
    """27 h into a stream the 33-bit clock has wrapped once; a fresh run must not
    stamp lines 26.5 h too early."""
    h = Harness(monkeypatch, tmp_path, start_time=START)
    h.clock.wall_at_zero = START + 27 * 3600 + 3
    raw = 27 * 3600 - TS_WRAP_SECONDS
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, raw) + ["end"]]
    items = await h.run()
    assert items[0].audio_start_time == pytest.approx(START + 27 * 3600 + 0.036)


async def test_a_clock_that_jumps_back_is_not_trusted(h, caplog):
    h.resolutions = [resolved()]
    # The sequence continues but the clock restarts near zero: a timeline reset.
    h.scripts = [run_of(100, 6, 4990.0) + run_of(106, 6, 10.0) + ["end"]]
    caplog.set_level(logging.WARNING)
    items = await h.run()
    assert [i.vod_accurate for i in items] == [True, False]
    assert items[1].audio_start_time == pytest.approx(h.clock.wall() - 6.0 - 1, abs=10)
    assert "not marked VOD-accurate" in caplog.text


async def test_a_clock_that_restarted_long_ago_is_not_trusted(monkeypatch, tmp_path, caplog):
    """Seen live on a 24/7 stream: the sequence numbers and the clock restarted
    days after the stream started, so start time + clock lands hours away from
    now. Those lines can't be VOD-accurate."""
    h = Harness(monkeypatch, tmp_path)
    h.clock.wall_at_zero = START + 6_534_178.0  # 75 days in
    raw = 59_700.0  # the clock counts from a restart ~4 days ago
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, raw) + ["end"]]
    caplog.set_level(logging.WARNING)
    items = await h.run()
    assert [i.vod_accurate for i in items] == [False]
    assert items[0].audio_start_time == pytest.approx(h.clock.wall() - 6.0 - 1, abs=10)
    assert items[0].lag_from is None  # stamped by arrival: lag measured from the audio edge
    assert "doesn't count from the stream's start" in caplog.text


async def test_an_outage_longer_than_the_start_time_check_allows_stays_vod_accurate(h):
    """Seen live: YouTube's clock, like the VOD, skips the time a stream is down,
    so after outages start + clock falls behind now by their total length. Lines
    after them are still exactly where the VOD has them."""
    outage = 2000.0  # well past lfs_gap_seconds + 300 s
    h.resolutions = [resolved(), resolved()]
    h.scripts = [run_of(100, 7, 4990.0) + [("sleep", outage)], run_of(107, 12, 4990.0 + 7 * SEG) + ["end"]]
    items = await h.run()
    assert [i.raw for i in items] == [b"TS:100-105", b"TS:106-111", b"TS:112-117", b"TS:118-118"]
    assert [i.vod_accurate for i in items] == [True] * 4
    assert items[1].audio_start_time == pytest.approx(START + 4990.0 + 6 * SEG + 0.036)
    assert h.stats.reconnects == 1


async def test_segments_lost_in_an_outage_stay_vod_accurate(h):
    h.resolutions = [resolved(), resolved()]
    h.scripts = [run_of(100, 7, 4990.0) + [("sleep", 2000.0)], run_of(400, 7, 4990.0 + 300 * SEG) + ["end"]]
    items = await h.run()
    assert all(i.vod_accurate for i in items)
    after = next(i for i in items if i.raw.startswith(b"TS:400-"))
    assert after.audio_start_time == pytest.approx(START + 4990.0 + 300 * SEG + 0.036)


async def test_a_clock_that_leaps_ahead_is_not_trusted(h):
    """Continuing a timeline means never running ahead of the time that passed."""
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 7, 4990.0) + run_of(107, 6, 90_000.0) + ["end"]]
    items = await h.run()
    assert [i.vod_accurate for i in items] == [True, True, False]


async def test_a_restarted_run_continues_the_last_runs_timeline(h, monkeypatch, tmp_path):
    """The capture restarted mid-stream, after outages the stream clock skipped:
    its first chunk carries on exactly where the last run's ended."""
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    assert [i.vod_accurate for i in await h.run()] == [True]
    h2 = Harness(monkeypatch, tmp_path)
    h2.clock.wall_at_zero = h.clock.wall() + 2000.0
    h2.resolutions = [resolved()]
    h2.scripts = [run_of(106, 6, 4990.0 + 6 * SEG) + ["end"]]
    items = await h2.run()
    assert [i.vod_accurate for i in items] == [True]
    assert items[0].audio_start_time == pytest.approx(START + 4990.0 + 6 * SEG + 0.036)
    # Not an exact continuation: checked against the start time as usual.
    h3 = Harness(monkeypatch, tmp_path)
    h3.clock.wall_at_zero = h2.clock.wall() + 2000.0
    h3.resolutions = [resolved()]
    h3.scripts = [run_of(150, 6, 4990.0 + 50 * SEG) + ["end"]]
    assert [i.vod_accurate for i in await h3.run()] == [False]


async def test_catching_up_a_backlog_stays_vod_accurate(h):
    """Ten minutes behind (after a restart) is still the stream's own clock."""
    h.clock.wall_at_zero = START + 5000.0 + 550.0
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    items = await h.run()
    assert [i.vod_accurate for i in items] == [True]


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


async def test_not_live_ends_without_a_connection(h):
    h.resolutions = [StreamNotLiveError("live_status is was_live")]
    assert await h.run() == []
    assert h.sources == []


async def test_members_only_ends(h):
    h.resolutions = [StreamUnavailableError("members-only")]
    assert await h.run() == []


async def test_resolve_failures_back_off(h):
    h.resolutions = [ResolveError("timed out"), ResolveError("timed out"), resolved()]
    h.scripts = [["end"]]
    await h.run()
    gaps = [b - a for a, b in zip(h.resolve_calls, h.resolve_calls[1:], strict=False)]
    assert gaps[0] >= 15 and gaps[1] >= 30


async def test_resolve_backoff_stays_under_half_the_catch_up_window(monkeypatch, tmp_path):
    """With lfs_gap_seconds at 300, waiting 300 s between failed yt-dlp runs could
    leave a stream that came back right after one more than 300 s behind by the
    next, and the source would skip everything since it came back."""
    h = Harness(monkeypatch, tmp_path, lfs_gap=300)
    h.resolutions = [ResolveError("got no usable formats")] * 6 + [resolved()]
    h.scripts = [["end"]]
    await h.run()
    gaps = [b - a for a, b in zip(h.resolve_calls, h.resolve_calls[1:], strict=False)]
    assert gaps == pytest.approx([15, 30, 60, 120, 150, 150], abs=1)


async def test_unsupported_is_logged_once_and_retried_rarely(h, caplog):
    caplog.set_level(logging.ERROR)
    h.resolutions = [UnsupportedSourceError("SABR only"), UnsupportedSourceError("SABR only"), StreamNotLiveError("ended")]
    await h.run()
    assert h.resolve_calls[1] - h.resolve_calls[0] >= 600
    assert caplog.text.count("cannot read this stream") == 1


async def test_unsupported_from_the_source_factory_is_retried_rarely(h, monkeypatch):
    calls = []

    def create(resolved_stream, resume, ctx):
        calls.append(h.clock.t)
        raise UnsupportedSourceError("no segment source for protocols ['sabr']")

    monkeypatch.setattr(ct, "create_segment_source", create)
    h.resolutions = [resolved(), resolved()]
    await h.run()
    assert len(calls) == 2 and calls[1] - calls[0] >= 600


async def test_resolve_stopped_ends(h):
    h.resolutions = [ResolveStopped("stopped while resolving")]
    assert await h.run() == []


async def test_expired_urls_reconnect_and_continue_at_the_same_sequence(h):
    h.resolutions = [resolved(), resolved()]
    h.scripts = [run_of(100, 4, 4990.0) + [SourceExpiredError("audio sequence 104: HTTP 403")], run_of(104, 2, 4990.0 + 4 * SEG) + ["end"]]
    items = await h.run()
    assert h.sources[1].resume == 104
    assert h.sources[0].closed
    # The chunk spans the reconnect: nothing lost, no gap.
    assert [i.raw for i in items] == [b"TS:100-105"]
    assert h.stats.reconnects == 1


async def test_idle_connection_reconnects_then_ends_when_the_stream_is_over(h):
    h.resolutions = [resolved()]  # the second resolve says not live
    h.scripts = [run_of(100, 2, 4990.0) + [None] * 30]
    await h.run()
    assert len(h.resolve_calls) == 2
    assert h.stats.reconnects == 1


async def test_frozen_playlist_does_not_end_the_run_while_the_stream_is_live(h):
    """A streamer outage: no segments for minutes, but yt-dlp still says live.
    The run keeps going (paced) instead of returning to the watcher."""
    h.resolutions = [resolved(), resolved(), resolved(), resolved()]
    h.scripts = [[None] * 20, [None] * 20, [None] * 20, run_of(100, 6, 4990.0) + ["end"]]
    items = await h.run()
    assert len(items) == 1
    assert len(h.resolve_calls) == 4
    # Fruitless reconnects back off.
    gaps = [b - a for a, b in zip(h.resolve_calls, h.resolve_calls[1:], strict=False)]
    assert gaps[1] > gaps[0]


async def test_nothing_usable_reconnects(h, caplog):
    """Segments keep coming (so the idle check never fires) but none has audio."""
    caplog.set_level(logging.INFO)
    h.resolutions = [resolved()]
    h.scripts = [[group(100 + i, 4990.0 + i * SEG, audio=False) for i in range(70)]]
    await h.run()
    assert h.stats.reconnects == 1
    assert len(h.resolve_calls) == 2
    assert "reconnecting: nothing usable in 61s" in caplog.text
    # The connection produced nothing, so the next yt-dlp run backs off 15 s.
    assert h.resolve_calls[1] == pytest.approx(61 + 15, abs=1)


async def test_repeated_source_errors_reconnect(h):
    h.resolutions = [resolved()]
    h.scripts = [[SourceError("audio playlist: Timeout")] * 3]
    await h.run()
    assert h.stats.reconnects == 1


def _clock_of(sequence):
    return 4990.0 + (sequence - 100) * SEG


async def _play(source, timeout):
    await asyncio.sleep(0)
    action = next(source.script, None)
    if action is None:
        source.clock.sleep(timeout)
        return None
    if action == "end":
        source._ended = True
        return None
    source.clock.sleep(1.0)
    source._next_sequence = action.sequence + 1
    return action


async def test_proactive_refresh_swaps_urls_without_a_gap(h, monkeypatch):
    expires = h.clock.wall() + 20 * 60 + 2.5
    h.resolutions = [resolved(expires_at=expires), resolved(expires_at=expires + 6 * 3600)]

    def create(resolved_stream, resume, ctx):
        if not h.sources:
            # A healthy connection: a new segment every second, indefinitely.
            script = (group(s, _clock_of(s)) for s in range(100, 10_000))
        else:
            script = iter([group(resume, _clock_of(resume)), group(resume + 1, _clock_of(resume + 1)), "end"])
        source = FakeSource(resolved_stream, resume, ctx, [], h.clock)
        source.script = script  # type: ignore[assignment]
        source.next_group = lambda timeout, src=source: _play(src, timeout)  # type: ignore[method-assign]
        h.sources.append(source)
        return source

    monkeypatch.setattr(ct, "create_segment_source", create)
    items = await h.run()
    assert len(h.resolve_calls) == 2
    swap_at = h.sources[1].resume
    assert swap_at == h.sources[0].checkpoint and swap_at > 100
    assert h.sources[0].closed
    assert h.stats.reconnects == 0  # a swap, not a reconnect
    # Every sequence from 100 to the end was queued exactly once, across the swap.
    covered = []
    for item in items:
        first, last = (int(x) for x in item.raw[3:].split(b"-"))
        covered += list(range(first, last + 1))
    assert covered == list(range(100, swap_at + 2))
    assert all(i.vod_accurate for i in items)


async def test_failed_proactive_refresh_keeps_the_current_urls_until_hard_expiry(h):
    expires = h.clock.wall() + 20 * 60 - 1
    h.resolutions = [resolved(expires_at=expires), ResolveError("timed out"), StreamNotLiveError("ended")]
    # Keep the connection alive with segments; then let time run out.
    h.scripts = [run_of(100, 6, 4990.0) + [("sleep", 19 * 60)] + run_of(106, 3, 4990.0 + 6 * SEG)]
    items = await h.run()
    assert [i.raw for i in items][0] == b"TS:100-105"
    # Refresh attempt failed, the connection carried on, then the hard expiry reconnected.
    assert len(h.resolve_calls) == 3
    assert h.stats.reconnects == 1


async def test_failing_refreshes_keep_the_connection_until_the_urls_are_about_to_expire(h, monkeypatch, caplog):
    """Segments keep flowing on the current URLs while every background refresh
    fails; only the hard expiry rebuilds the connection."""
    caplog.set_level(logging.INFO)
    expires = h.clock.wall() + 15 * 60 + 1  # just enough headroom to be trusted
    calls: list[float] = []

    async def resolve(config, url, media_type, should_stop, raw_log=None, timeout=90.0):
        calls.append(h.clock.t)
        if len(calls) == 1:
            return resolved(expires_at=expires)
        if h.clock.t < 15 * 60 + 1 - 60:
            raise ResolveError("timed out")
        raise StreamNotLiveError("ended")  # the reconnect finds the stream over

    monkeypatch.setattr(ct, "resolve_stream", resolve)
    h.scripts = [run_of(100, 2000, 4990.0)]
    items = await h.run()
    assert "reconnecting: the stream URLs are about to expire" in caplog.text
    assert h.stats.reconnects == 1
    # 842 s of segments on the old URLs (140 chunks, then a 2-segment leftover flushed at the end), none lost.
    assert len(items) == 141 and [i.raw for i in items[-2:]] == [b"TS:934-939", b"TS:940-941"]
    refreshes = calls[1:-1]
    assert len(refreshes) >= 10
    assert all(b - a >= TimelineCapture._PROACTIVE_RETRY_SECONDS for a, b in zip(refreshes, refreshes[1:], strict=False))


async def test_an_expiry_too_close_to_trust_is_ignored(h, caplog):
    caplog.set_level(logging.WARNING)
    h.resolutions = [resolved(expires_at=h.clock.wall() + 60)]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    await h.run()
    assert len(h.resolve_calls) == 1
    assert "ignoring that" in caplog.text


async def test_reset_follows_the_new_numbering_on_the_same_urls(h):
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 3, 4990.0) + [SourceResetError("sequence numbers went back")], run_of(5, 6, 4.0) + ["end"]]
    items = await h.run()
    # Same URLs, no yt-dlp run, read from the start of the new numbering (the source clamps it).
    assert len(h.resolve_calls) == 1
    assert h.sources[1].resume == 0
    assert h.sources[1].resolved is h.sources[0].resolved
    # The partial chunk before the reset is flushed, then the new timeline starts fresh.
    assert [i.raw for i in items][:2] == [b"TS:100-102", b"TS:5-10"]


async def test_missing_video_triggers_a_rate_limited_reconnect(h, monkeypatch):
    h.resolutions = [resolved(), resolved()]

    class VideoGone(FakeSource):
        async def next_group(self, timeout):
            g = await super().next_group(timeout)
            if g is not None:
                self.consecutive_missing[TRACK_VIDEO] = 40
            return g

    created = []

    def create(resolved_stream, resume, ctx):
        script = run_of(100, 6, 4990.0) if not created else run_of(106, 6, 4990.0 + 6 * SEG) + ["end"]
        source = VideoGone(resolved_stream, resume, ctx, script, h.clock)
        created.append(source)
        return source

    monkeypatch.setattr(ct, "create_segment_source", create)
    await h.run()
    # One reconnect for the missing video; the second connection isn't cut short again.
    assert len(created) == 2 and h.stats.reconnects == 1


async def test_stop_flushes_the_partial_chunk_and_exits(h, caplog):
    caplog.set_level(logging.INFO)
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 3, 4990.0) + [None] * 100]
    h.stop_after = 10.0
    items = await h.run()
    assert [i.raw for i in items] == [b"TS:100-102"]
    assert len(h.resolve_calls) == 1
    assert "stopped on request" in caplog.text


async def test_unexpected_errors_are_contained(h, caplog):
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 2, 4990.0) + [RuntimeError("boom")]]
    caplog.set_level(logging.ERROR)
    items = await h.run()  # must not raise
    assert [i.raw for i in items] == [b"TS:100-101"]
    assert "unexpected error: boom" in caplog.text
    assert h.sources[0].closed


async def test_mux_failures_are_not_queued(h, monkeypatch):
    async def mux(ffmpeg_path, chunk, workdir, output, timeout=30.0):
        raise MuxError("ffmpeg exit=1")

    monkeypatch.setattr(ct, "mux_chunk", mux)
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    assert await h.run() == []
    assert h.stats.chunks_failed == 1


async def test_run_log_records_the_outcome(h):
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    await h.run()
    log = h.run_log
    assert "YouTube timeline started" in log and "(stream: abcdefghijk)" in log
    assert "resolved video=video, audio=audio" in log
    assert "stopped at" in log and "the stream ended" in log and "chunks=1" in log
    assert list(h.state.segments_dir.iterdir()) == []  # work dir cleaned up


async def test_a_slow_url_refresh_does_not_block_segments(h, monkeypatch):
    """The refresh runs yt-dlp in a background task; segments keep being
    consumed meanwhile and nothing is torn down."""
    expires = h.clock.wall() + 20 * 60 - 1
    release = asyncio.Event()
    in_refresh = asyncio.Event()
    calls = []

    async def resolve(config, url, media_type, should_stop, raw_log=None, timeout=90.0):
        calls.append(h.clock.t)
        if len(calls) == 1:
            return resolved(expires_at=expires)
        in_refresh.set()
        await asyncio.wait_for(release.wait(), 10)
        return resolved(expires_at=expires + 6 * 3600)

    monkeypatch.setattr(ct, "resolve_stream", resolve)
    consumed_during_refresh = []

    def create(resolved_stream, resume, ctx):
        source = FakeSource(resolved_stream, resume, ctx, [], h.clock)
        first = not h.sources
        h.sources.append(source)

        state = {"s": resume or 100}

        async def play(timeout, src=source, state=state):
            await asyncio.sleep(0)
            if not first:
                if state["s"] > (resume or 100) + 1:
                    src._ended = True
                    return None
            elif in_refresh.is_set():
                consumed_during_refresh.append(state["s"])
                if len(consumed_during_refresh) >= 70:  # 70 s of segments while yt-dlp hangs
                    release.set()
            seq = state["s"]
            state["s"] += 1
            h.clock.sleep(1.0)
            src._next_sequence = seq + 1
            return group(seq, 4990.0 + (seq - 100) * SEG)

        source.next_group = play  # type: ignore[method-assign]
        return source

    monkeypatch.setattr(ct, "create_segment_source", create)
    await h.run()
    assert len(consumed_during_refresh) >= 70
    assert h.stats.reconnects == 0  # not torn down as "nothing usable"
    assert len(h.sources) == 2 and h.sources[1].resume == h.sources[0].checkpoint


async def test_unexpected_mux_errors_are_contained(h, monkeypatch):
    async def mux(ffmpeg_path, chunk, workdir, output, timeout=30.0):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(ct, "mux_chunk", mux)
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 9, 4990.0) + ["end"]]
    assert await h.run() == []  # no exception escapes run(), even from the final flush
    assert h.stats.chunks_failed == 2


@pytest.mark.parametrize("content", ["[]", "null", "5", '"x"', "{bad json", '{"stream_id": "abcdefghijk", "next_sequence": true}'])
async def test_a_corrupt_state_file_is_ignored(h, content):
    (h.state.root / "youtube_timeline_state.json").write_text(content)
    h.resolutions = [resolved()]
    await h.run()
    assert h.sources[0].resume is None


async def test_the_run_log_stays_bounded_during_long_runs(h):
    """The reference trimmed its run log at most every 5 minutes; a TailLog
    bounds itself on every append, yt-dlp's resolve output included."""
    log = h.capture._run_log
    log.max_bytes = 4096
    h.ytdlp_stderr = "[youtube] abcdefghijk: Downloading webpage\n" * 100  # ~4.3 KB per resolve
    h.resolutions = [resolved(), resolved()]
    h.scripts = [[None] * 20, ["end"]]
    await h.run()
    text = h.run_log
    assert log.path.stat().st_size <= 4096
    assert "Downloading webpage" in text  # yt-dlp's stderr goes to the run log
    assert text.rstrip().splitlines()[-1].startswith("--- YouTube timeline stopped at")


async def test_yt_dlp_never_runs_more_often_than_the_floor(h):
    """Even productive connections that drop quickly don't re-run yt-dlp faster
    than every 15 s."""
    h.resolutions = [resolved(), resolved(), resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + [SourceExpiredError("403")], run_of(106, 1, 4990.0 + 6 * SEG) + [SourceExpiredError("403")]]
    await h.run()
    gaps = [b - a for a, b in zip(h.resolve_calls, h.resolve_calls[1:], strict=False)]
    assert len(gaps) == 2 and all(g >= TimelineCapture._MIN_RESOLVE_INTERVAL_SECONDS for g in gaps)


async def test_failed_url_refreshes_are_retried_every_minute_not_continuously(h, monkeypatch):
    expires = h.clock.wall() + 20 * 60 - 1
    calls = []

    async def resolve(config, url, media_type, should_stop, raw_log=None, timeout=90.0):
        calls.append(h.clock.t)
        if len(calls) == 1:
            return resolved(expires_at=expires)
        raise ResolveError("timed out")

    monkeypatch.setattr(ct, "resolve_stream", resolve)

    def create(resolved_stream, resume, ctx):
        script = [group(s, 4990.0 + (s - 100) * SEG) for s in range(100, 400)] + ["end"]
        source = FakeSource(resolved_stream, resume, ctx, script, h.clock)
        h.sources.append(source)
        return source

    monkeypatch.setattr(ct, "create_segment_source", create)
    await h.run()
    refreshes = calls[1:]
    assert len(refreshes) >= 3
    assert all(b - a >= TimelineCapture._PROACTIVE_RETRY_SECONDS for a, b in zip(refreshes, refreshes[1:], strict=False))


def _chunk_ending_at(clock_end, duration=6.0):
    return TimelineChunk(1, 6, clock_end - duration, duration)


def test_clock_trust_bounds(h):
    """start + clock may run up to 900 s ahead of now, or behind by the catch-up
    window plus 300 s of delivery delay."""
    cap = h.capture
    cap._start_time = START
    now = START + 10_000.0
    ahead = TimelineCapture._CLOCK_AHEAD_TOLERANCE_SECONDS
    behind = h.config.server.stale_threshold.lfs_gap_seconds + TimelineCapture._CLOCK_BEHIND_TOLERANCE_SECONDS
    assert cap._clock_trusted(_chunk_ending_at(10_000.0 + ahead - 1), now)
    assert not cap._clock_trusted(_chunk_ending_at(10_000.0 + ahead + 1), now)
    assert cap._clock_trusted(_chunk_ending_at(10_000.0 - behind + 1), now)
    assert not cap._clock_trusted(_chunk_ending_at(10_000.0 - behind - 1), now)
    # A normal live chunk: a few seconds behind.
    assert cap._clock_trusted(_chunk_ending_at(10_000.0 - 4), now)


async def test_the_real_sleep_returns_promptly_on_stop(tmp_path):
    config = make_config(tmp_path)
    capture = TimelineCapture(config, config.streamers[0], ChannelState("chan", tmp_path / "tmp" / "chan"))
    stopped = {"set": False}
    capture._should_stop = lambda: stopped["set"]
    asyncio.get_running_loop().call_later(0.2, stopped.__setitem__, "set", True)
    started = time.monotonic()
    await capture._sleep(30)
    assert time.monotonic() - started < 1.5


# ----------------------------------------------------------------------
# Cloud worker: chunk files, pipeline hand-off, cancellation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("media_type", [MediaType.VIDEO, MediaType.AUDIO])
async def test_emitted_chunks_carry_their_file_pcm_and_clock_stamp(monkeypatch, tmp_path, media_type):
    h = Harness(monkeypatch, tmp_path, media_type=media_type)
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    h.pcm_seconds = 5.98
    existed = []
    original = h.emit

    async def emit(chunk):
        existed.append(chunk.path.is_file())
        await original(chunk)

    h.emit = emit  # type: ignore[method-assign]
    (item,) = await h.run()
    chunk = item.chunk
    assert existed == [True]
    work_dir = chunk.path.parent
    assert work_dir.parent == h.state.segments_dir and work_dir.name.startswith("t")
    assert chunk.path.name == "chunk000000.ts"
    assert h.mux_calls == [(h.config.capture.ffmpeg_path, work_dir / "mux", chunk.path)]
    assert (chunk.key, chunk.stream_id, chunk.media_type) == ("key", "abcdefghijk", media_type)
    # The decoded audio and its measured duration, not the playlist's.
    assert chunk.pcm == b"\x00\x00" * round(16000 * 5.98)
    assert chunk.duration == pytest.approx(5.98)
    assert chunk.vod_accurate is True
    assert chunk.audio_start_time == pytest.approx(START + 4990.0 + FakeSource.vod_offset)
    assert h.stats == TimelineStats(chunks_produced=1)


async def test_chunks_shorter_than_min_chunk_seconds_are_dropped_and_deleted(h, caplog):
    caplog.set_level(logging.WARNING)
    h.keep_files = True
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 18, 4990.0) + ["end"]]
    h.pcm_seconds = [0.3, 6.0, 0.2]
    items = await h.run()
    assert [i.raw for i in items] == [b"TS:106-111"]
    assert h.stats.chunks_failed == 2 and h.stats.chunks_produced == 1
    # Only the emitted chunk's file is left, for the pipeline.
    (work_dir,) = h.state.segments_dir.iterdir()
    assert [p.name for p in work_dir.iterdir()] == [items[0].path.name]
    # A rate-limited warning.
    assert caplog.text.count("under transcription.min_chunk_seconds") == 1


@pytest.mark.parametrize("failure", ["error", "empty"])
async def test_undecodable_chunks_are_dropped_and_deleted(h, monkeypatch, caplog, failure):
    caplog.set_level(logging.WARNING)
    paths = []

    async def extract(ffmpeg_path, path, timeout=30.0):
        paths.append(path)
        if failure == "error":
            raise AudioExtractError("ffmpeg failed on chunk000000.ts (rc=1)")
        return b""

    monkeypatch.setattr(ct.audio, "extract_pcm", extract)
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    assert await h.run() == []
    assert h.stats.chunks_failed == 1
    assert not paths[0].exists()
    assert "have no decodable audio" in caplog.text
    assert not (h.state.root / "youtube_timeline_state.json").exists()


async def test_unexpected_decode_errors_are_contained(h, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    calls = []

    async def extract(ffmpeg_path, path, timeout=30.0):
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("decoder on fire")
        return b"\x00\x00" * 16000 * 6

    monkeypatch.setattr(ct.audio, "extract_pcm", extract)
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 12, 4990.0) + ["end"]]
    items = await h.run()
    assert [i.raw for i in items] == [b"TS:106-111"]
    assert h.stats.chunks_failed == 1
    assert not calls[0].exists()
    assert "decoding failed unexpectedly" in caplog.text


async def test_a_slow_emit_does_not_break_the_run(h):
    """The pipeline's queue is bounded, so emit() may hold the capture while
    transcription catches up. That wait isn't the connection going idle: no
    reconnect, nothing lost."""
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 18, 4990.0) + [None, None] + run_of(118, 6, 4990.0 + 18 * SEG) + ["end"]]
    original = h.emit

    async def slow_emit(chunk):
        await h.clock.asleep(45.0)  # longer than the idle reconnect
        await original(chunk)

    h.emit = slow_emit  # type: ignore[method-assign]
    items = await h.run()
    assert [i.raw for i in items] == [b"TS:100-105", b"TS:106-111", b"TS:112-117", b"TS:118-123"]
    assert h.stats.reconnects == 0 and len(h.resolve_calls) == 1
    assert all(i.vod_accurate for i in items)
    assert [i.audio_start_time for i in items] == pytest.approx([START + 4990.0 + n * SEG + 0.036 for n in (0, 6, 12, 18)])


async def test_pipeline_backpressure_widens_the_catch_up_window(monkeypatch, tmp_path, caplog):
    """The source skips to the live edge past lfs_gap_seconds behind it, which is
    for hiccups and restarts. Time spent waiting for the pipeline (emit) widens
    that window, so a backed-up pipeline never costs segments YouTube still
    serves, and the window shrinks back once the source catches up."""
    h = Harness(monkeypatch, tmp_path, lfs_gap=60.0)
    live = {"head": 117}
    h.live_head = lambda: live["head"]
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 18, 4990.0) + ["end"]]
    windows: list[float] = []
    original = h.emit

    async def emit(chunk):
        windows.append(h.sources[0].ctx.max_lag_seconds)
        if chunk.path.read_bytes() == b"TS:100-105":
            live["head"] += 100  # the pipeline holds the capture for 100 s
            await h.clock.asleep(100.0)
        else:
            live["head"] = 117  # caught up again
        await original(chunk)

    h.emit = emit  # type: ignore[method-assign]
    caplog.set_level(logging.INFO)
    items = await h.run()
    assert [i.raw for i in items] == [b"TS:100-105", b"TS:106-111", b"TS:112-117"]
    # Widened by twice the 100 s wait right after it, back to lfs_gap once within it again.
    assert windows[0] == 60.0
    assert windows[1] == pytest.approx(60.0 + 2 * 100.0)
    assert h.sources[0].ctx.max_lag_seconds == 60.0
    assert "the pipeline is backed up" in caplog.text
    assert "caught up with the pipeline backlog" in caplog.text


def yt_resolved(yt, tracks=(TRACK_VIDEO, TRACK_AUDIO)):
    return ResolvedStream(
        "abcdefghijk", "is_live", tuple(ResolvedTrack(t, "m3u8_native", yt.playlist_url(t), yt.itag(t)) for t in tracks), None
    )


def use_real_source(monkeypatch, h, yt):
    """The capture reads a real HlsSegmentSource against the fake YouTube, on the harness's fake clock."""

    async def wait_slice(task, seconds):
        # Let ready fetch tasks run; fake time only passes for one still pending.
        for _ in range(5):
            if task.done():
                return
            await asyncio.sleep(0)
        if not task.done():
            h.clock.sleep(seconds)

    if not h.resolutions:
        h.resolutions = [yt_resolved(yt)] * 5

    def create(resolved_stream, resume, ctx):
        source = HlsSegmentSource(resolved_stream, resume, ctx, client_factory=yt.client)
        source._monotonic = h.clock.mono  # type: ignore[method-assign]
        source._sleep = h.clock.asleep  # type: ignore[method-assign]
        source._wait_slice = wait_slice  # type: ignore[method-assign]
        return source

    monkeypatch.setattr(ct, "create_segment_source", create)


async def test_a_backed_up_pipeline_loses_nothing_with_the_real_source(monkeypatch, tmp_path):
    """Regression: the pipeline takes 15 s per 6 s chunk (the server is down and
    every line retries), so the capture falls further behind live with every
    chunk, far past lfs_gap_seconds. The reference's unbounded queue never held
    its capture, and live-edge keeps writing segments to disk, so neither loses
    anything; the timeline must not skip either."""
    h = Harness(monkeypatch, tmp_path, lfs_gap=60.0)
    yt = FakeYouTube(head=10)
    h.clock.wall_at_zero = START + yt_clock(10) + 3.0
    last = 10 + 240

    def tick():
        for track in yt.head:
            yt.head[track] = min(last, yt.head[track] + 1)
        if yt.head[TRACK_AUDIO] >= last:
            yt.ended = True

    h.clock.on_second = tick
    use_real_source(monkeypatch, h, yt)
    original = h.emit

    async def emit(chunk):
        await h.clock.asleep(15.0)
        await original(chunk)

    h.emit = emit  # type: ignore[method-assign]
    items = await h.run()
    covered: set[int] = set()
    for item in items:
        first, last_seq = item.raw.decode().split(":")[1].split("-")
        covered.update(range(int(first), int(last_seq) + 1))
    assert h.stats.segments_skipped == 0
    assert covered == set(range(10, last + 1))
    assert all(i.vod_accurate for i in items)
    assert "behind the live edge" not in h.run_log


async def test_a_flush_that_waits_on_the_pipeline_widens_the_window_before_the_next_source(monkeypatch, tmp_path):
    """The numbering restarts while the pipeline is backed up: the reset's flush of
    the partial chunk waits 100 s in emit(). The next source's very first lag check
    must already see that wait, or it skips the new numbering's segments YouTube
    still serves."""
    h = Harness(monkeypatch, tmp_path, lfs_gap=60.0)
    yt = FakeYouTube(head=10)
    h.clock.wall_at_zero = START + yt_clock(10) + 3.0
    st = {"restarted": False, "in_flush": False, "blocked": False}

    def tick():
        if not st["restarted"] and h.clock.t >= 40.0:
            st["restarted"] = True
            for track in yt.head:
                yt.head[track] = 5  # the new numbering
            return
        for track in yt.head:
            yt.head[track] += 1
        if st["restarted"] and yt.head[TRACK_AUDIO] >= 200:
            yt.ended = True

    h.clock.on_second = tick
    use_real_source(monkeypatch, h, yt)
    original_flush = h.capture._flush

    async def flush(info, assembler):
        st["in_flush"] = True
        try:
            await original_flush(info, assembler)
        finally:
            st["in_flush"] = False

    h.capture._flush = flush  # type: ignore[method-assign]
    original = h.emit

    async def emit(chunk):
        if st["in_flush"] and st["restarted"] and not st["blocked"]:
            st["blocked"] = True
            await h.clock.asleep(100.0)  # the pipeline holds the reset's flush
        await original(chunk)

    h.emit = emit  # type: ignore[method-assign]
    await h.run()
    assert st["blocked"] and "numbering restarted" in h.run_log
    assert "behind the live edge" not in h.run_log
    assert h.stats.segments_skipped == 0


async def test_falling_behind_without_waiting_on_the_pipeline_is_not_blamed_on_it(monkeypatch, tmp_path):
    """Downloads slower than real time make the source fall past lfs_gap and skip,
    as they should; emit() never waited (a put into a queue with room still takes
    a microsecond), so nothing may say the pipeline is backed up."""
    h = Harness(monkeypatch, tmp_path, lfs_gap=20.0)
    yt = FakeYouTube(head=10)
    h.clock.wall_at_zero = START + yt_clock(10) + 3.0
    last = 10 + 150

    def tick():
        for track in yt.head:
            yt.head[track] = min(last, yt.head[track] + 1)
        if yt.head[TRACK_AUDIO] >= last:
            yt.ended = True

    h.clock.on_second = tick
    yt.on_chunk = lambda: h.clock.sleep(0.6)  # 0.6 s per segment download: 1.2 s per 1 s sequence
    use_real_source(monkeypatch, h, yt)
    original = h.emit

    async def emit(chunk):
        h.clock.sleep(1e-6)
        await original(chunk)

    h.emit = emit  # type: ignore[method-assign]
    await h.run()
    assert "behind the live edge, skipping" in h.run_log
    assert "the pipeline is backed up" not in h.run_log
    assert "pipeline backlog" not in h.run_log


async def test_catching_up_a_widened_backlog_keeps_trusting_the_clock(monkeypatch, tmp_path):
    """The pipeline holds the capture 500 s (nothing skipped), then a reconnect
    resolves a muxed-only format: the clock track changes and the anchor goes. The
    backlog that follows is checked against the start time, with the widened
    window as the catch-up allowance: still VOD-accurate, not stamped by arrival."""
    h = Harness(monkeypatch, tmp_path, lfs_gap=60.0)
    yt = FakeYouTube(head=10, tracks=(TRACK_VIDEO, TRACK_AUDIO, TRACK_MUXED))
    h.clock.wall_at_zero = START + yt_clock(10) + 3.0
    last = 10 + 600

    def tick():
        for track in yt.head:
            yt.head[track] = min(last, yt.head[track] + 1)
        if yt.head[TRACK_AUDIO] >= last:
            yt.ended = True

    h.clock.on_second = tick
    h.resolutions = [yt_resolved(yt), yt_resolved(yt, (TRACK_MUXED,))]
    use_real_source(monkeypatch, h, yt)
    yt.status["/itag/234/playlist/index.m3u8/sq/30/"] = [403]  # URLs rejected: reconnect at sequence 30

    def intercept(url):
        if "/itag/95/" in url and "/sq/" in url:
            sq = int(url.split("/sq/")[1].split("/")[0])
            if sq > yt.head[TRACK_MUXED]:
                return httpx.Response(404)
            return httpx.Response(200, content=ts_pes_packet(0x101, 0xC0, yt_clock(sq)) + video_segment(yt_clock(sq) + 0.01))
        return None

    yt.intercept = intercept
    original = h.emit
    blocked = {"done": False}

    async def emit(chunk):
        if not blocked["done"]:
            blocked["done"] = True
            await h.clock.asleep(500.0)
        await original(chunk)

    h.emit = emit  # type: ignore[method-assign]
    items = await h.run()
    after = [i for i in items if int(i.raw.decode().split(":")[1].split("-")[0]) >= 30]
    assert after, "no chunk after the reconnect"
    assert all(i.vod_accurate for i in after)
    for item in after:
        first = int(item.raw.decode().split(":")[1].split("-")[0])
        assert item.audio_start_time == pytest.approx(START + yt_clock(first) + HLS_CLOCK_TO_VOD_SECONDS)
    assert h.stats.segments_skipped == 0


async def test_an_estimated_start_counts_the_backlog_a_resumed_run_catches_up(monkeypatch, tmp_path):
    """No start time, and a run that resumes 300 s behind live: the start is
    estimated from a catch-up chunk, so the stream published since it has to be
    counted, or every later line lands 300 s in the future."""
    h = Harness(monkeypatch, tmp_path, start_time=None)
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 6, 4990.0) + ["end"]]
    await h.run()

    backlog = 300
    h2 = Harness(monkeypatch, tmp_path, start_time=None)
    h2.clock.wall_at_zero = h.clock.wall() + backlog
    # The live edge: 300 sequences past the resume point, then 1 per second of fake time.
    h2.live_head = lambda: 105 + backlog + int(h2.clock.t)
    h2.resolutions = [resolved()]
    catch_up = run_of(106, backlog, 4990.0 + 6 * SEG)
    live = run_of(106 + backlog, 30, 4990.0 + (6 + backlog) * SEG)
    h2.scripts = [catch_up + live + ["end"]]
    h2.steps = [0.05]
    items = await h2.run()
    assert h2.sources[0].resume == 106
    # The truth in this model: sequence 405, the live edge at h2's start, had just been
    # delivered (the assumed 2 s) when h2 started.
    true_start = h2.clock.wall_at_zero - 2.0 - (4990.0 + 306 * SEG)
    assert h2.capture._start_time == pytest.approx(true_start, abs=2)  # without the backlog: 294 s off
    assert not any(i.vod_accurate for i in items)


async def test_cancellation_mid_emit_cleans_up_and_propagates(h, monkeypatch):
    """Cancelled while the pipeline holds emit(): no final chunk, the source
    closed, the background refresh stopped, the unclaimed chunk file deleted,
    and the CancelledError re-raised."""
    expires = h.clock.wall() + 20 * 60 - 1
    refresh_cancelled = asyncio.Event()
    calls = []

    async def resolve(config, url, media_type, should_stop, raw_log=None, timeout=90.0):
        calls.append(h.clock.t)
        if len(calls) == 1:
            return resolved(expires_at=expires)
        try:
            await asyncio.Event().wait()  # yt-dlp hangs
        except asyncio.CancelledError:
            refresh_cancelled.set()
            raise
        raise AssertionError("unreachable")

    monkeypatch.setattr(ct, "resolve_stream", resolve)
    h.scripts = [run_of(100, 40, 4990.0)]
    blocked = asyncio.Event()
    held: list[Path] = []
    original = h.emit

    async def emit(chunk):
        if len(h.items) < 3:
            await original(chunk)
            return
        held.append(chunk.path)
        blocked.set()
        await asyncio.Event().wait()  # the pipeline never takes it

    h.emit = emit  # type: ignore[method-assign]
    task = asyncio.create_task(h.run())
    await asyncio.wait_for(blocked.wait(), 5)
    assert len(calls) == 2  # the refresh (started at 15 s) is under way
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert refresh_cancelled.is_set()
    assert h.sources[0].closed
    assert not held[0].exists()
    assert len(h.items) == 3  # no final chunk
    assert list(h.state.segments_dir.iterdir()) == []
    assert "stopped at" in h.run_log and ": cancelled (" in h.run_log


async def test_cancellation_while_waiting_for_segments_emits_no_final_chunk(h, monkeypatch):
    """A partial chunk is pending when the run is cancelled: it is dropped, not
    muxed and emitted (an emit could block a task being torn down)."""
    waiting = asyncio.Event()

    class Hanging(FakeSource):
        async def next_group(self, timeout):
            if not self.script:
                waiting.set()
                await asyncio.Event().wait()  # the network never answers
            return await super().next_group(timeout)

    created = []

    def create(resolved_stream, resume, ctx):
        source = Hanging(resolved_stream, resume, ctx, run_of(100, 3, 4990.0), h.clock)
        created.append(source)
        return source

    monkeypatch.setattr(ct, "create_segment_source", create)
    h.resolutions = [resolved()]
    task = asyncio.create_task(h.run())
    await asyncio.wait_for(waiting.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.items == [] and h.mux_calls == []
    assert created[0].closed
    assert list(h.state.segments_dir.iterdir()) == []
    assert ": cancelled (chunks=0," in h.run_log


async def test_the_work_dir_is_removed_once_the_pipeline_took_every_chunk(h):
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 13, 4990.0) + ["end"]]
    assert len(await h.run()) == 3
    assert list(h.state.segments_dir.iterdir()) == []


async def test_chunk_files_the_pipeline_still_holds_are_left_in_place(h):
    """The pipeline deletes (or queues) each file itself; a work dir it hasn't
    emptied yet stays for the boot sweep. The mux scratch dir always goes."""
    h.keep_files = True
    h.resolutions = [resolved()]
    h.scripts = [run_of(100, 12, 4990.0) + ["end"]]
    await h.run()
    (work_dir,) = h.state.segments_dir.iterdir()
    assert sorted(p.name for p in work_dir.iterdir()) == ["chunk000000.ts", "chunk000001.ts"]


async def test_yt_dlp_pacing_survives_a_new_run_of_the_same_stream(h, monkeypatch):
    """The watcher keeps one TimelineCapture per channel: a run that ends and
    is started again for the same stream still waits out the yt-dlp floor. A
    different stream starts fresh."""
    h.resolutions = [resolved(), resolved(), resolved()]
    h.scripts = [["end"], ["end"], ["end"]]
    await h.run()
    await h.run()
    assert h.resolve_calls == [0, pytest.approx(15)]
    h.info = StreamInfo(url="https://www.youtube.com/@chan/live", stream_id="zzzzzzzzzzz", is_live=True, start_time=START)
    monkeypatch.setattr(ct, "watch_url", lambda url, stream_id: WATCH_URL)
    await h.run()
    assert h.resolve_calls[2] == h.resolve_calls[1]


# ----------------------------------------------------------------------
# End to end: real HlsSegmentSource, real ffmpeg
# ----------------------------------------------------------------------

E2E_FIRST, E2E_COUNT = 10, 14  # the fake stream's sequences that carry real media
TS_MUXER_DELAY = 1.4  # the MPEG-TS muxer shifts every stream by this, -copyts or not


def _ffmpeg(*args):
    assert ffmpeg is not None
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def hls_media(tmp_path_factory):
    """Real media for the fake stream's sequences: 1 s MPEG-TS video segments
    whose PTS run on its clock, and 43-frame AAC groups (one HLS audio segment
    each) that get their sequence's ID3 timestamp when served."""
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("hls")
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=10:duration={E2E_COUNT}", "-c:v", "mpeg2video", "-g", "10",
        "-output_ts_offset", str(yt_clock(E2E_FIRST) - TS_MUXER_DELAY),
        "-f", "segment", "-segment_time", "1", "-segment_time_delta", "0.05", "-segment_format", "mpegts", str(d / "v%03d.ts"),
    )  # fmt: skip
    video = [(d / f"v{i:03d}.ts").read_bytes() for i in range(E2E_COUNT)]
    _ffmpeg(
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={E2E_COUNT + 1}", "-c:a", "aac", "-f", "adts", str(d / "a.aac")
    )
    data = (d / "a.aac").read_bytes()
    frames, pos = [], 0
    while pos + 7 <= len(data):
        length = ((data[pos + 3] & 0x03) << 11) | (data[pos + 4] << 3) | (data[pos + 5] >> 5)
        frames.append(data[pos : pos + length])
        pos += length
    audio = [b"".join(frames[i * 43 : (i + 1) * 43]) for i in range(E2E_COUNT)]
    assert all(len(a) and v for a, v in zip(audio, video, strict=True))
    return video, audio


@needs_ffmpeg
async def test_end_to_end_with_the_real_hls_source(monkeypatch, tmp_path, hls_media):
    """Drives the real HlsSegmentSource (against a fake YouTube) and the real
    ffmpeg mux and decode through the capture: lines are stamped start + YouTube
    clock + the HLS VOD offset."""
    video, audio = hls_media
    h = Harness(monkeypatch, tmp_path, fake_media=False)
    yt = FakeYouTube(head=E2E_FIRST)

    def serve(url):
        if "/sq/" not in url:
            return None  # playlists: the fake YouTube's own
        sq = int(url.split("/sq/")[1].split("/")[0])
        track = TRACK_VIDEO if f"/itag/{yt.itag(TRACK_VIDEO)}/" in url else TRACK_AUDIO
        index = sq - E2E_FIRST
        if sq > yt.head[track] or not 0 <= index < E2E_COUNT:
            return httpx.Response(404)
        body = video[index] if track == TRACK_VIDEO else id3_timestamp_tag(yt_clock(sq)) + audio[index]
        return httpx.Response(200, content=body)

    yt.intercept = serve
    # Wall time just after segment 10 of the fake stream, so the clock is anchored.
    h.clock.wall_at_zero = START + yt_clock(E2E_FIRST) + 3.0

    def tick():
        for track in yt.head:
            yt.head[track] += 1
        if yt.head[TRACK_AUDIO] >= 22:
            yt.ended = True

    h.clock.on_second = tick

    async def wait_slice(task, seconds):
        # Let ready fetch tasks run; fake time only passes for one still pending.
        for _ in range(5):
            if task.done():
                return
            await asyncio.sleep(0)
        if not task.done():
            h.clock.sleep(seconds)

    h.resolutions = [
        ResolvedStream(
            "abcdefghijk",
            "is_live",
            tuple(ResolvedTrack(t, "m3u8_native", yt.playlist_url(t), yt.itag(t)) for t in (TRACK_VIDEO, TRACK_AUDIO)),
            None,
        )
    ]
    sources: list[HlsSegmentSource] = []

    def create(resolved_stream, resume, ctx):
        src = HlsSegmentSource(resolved_stream, resume, ctx, client_factory=yt.client)
        src._monotonic = h.clock.mono  # type: ignore[method-assign]
        src._sleep = h.clock.asleep  # type: ignore[method-assign]
        src._wait_slice = wait_slice  # type: ignore[method-assign]
        sources.append(src)
        return src

    monkeypatch.setattr(ct, "create_segment_source", create)
    items = await h.run()
    assert len(items) == 3  # sequences 10-15, 16-21, then 22 flushed when the playlist ended
    first = items[0]
    assert first.vod_accurate is True
    assert first.audio_start_time == pytest.approx(START + yt_clock(10) + HLS_CLOCK_TO_VOD_SECONDS, abs=1e-4)
    assert items[1].audio_start_time == pytest.approx(START + yt_clock(16) + HLS_CLOCK_TO_VOD_SECONDS, abs=1e-4)
    assert HLS_CLOCK_TO_VOD_SECONDS == 0.036  # measured against a finished VOD
    # The real decode measured six HLS audio segments' worth of audio...
    assert first.duration == pytest.approx(6 * SEG, abs=0.05)
    assert len(first.pcm) == round(first.duration * 32000)
    # ...from a real mux that kept YouTube's clock on both tracks (-copyts).
    ts = scan_ts_pts(first.raw)
    assert ts is not None and ts.audio_pts is not None and ts.video_pts is not None
    assert ts.audio_pts == pytest.approx(yt_clock(10) + TS_MUXER_DELAY, abs=0.01)
    assert ts.video_pts == pytest.approx(yt_clock(10) + TS_MUXER_DELAY, abs=0.05)
    assert h.stats.chunks_produced == 3 and h.stats.chunks_failed == 0
    assert all(c.is_closed for c in yt.clients)
    assert list(h.state.segments_dir.iterdir()) == []
    state = json.loads((h.state.root / "youtube_timeline_state.json").read_text())
    assert state["next_sequence"] == 23
