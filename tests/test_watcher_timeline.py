"""Watcher strategy routing: the YouTube timeline vs live-edge capture."""

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import make_config

from live_transcript_cloud_worker import watcher as watcher_module
from live_transcript_cloud_worker.capture_timeline import TimelineCapture, TimelineStats
from live_transcript_cloud_worker.config import StreamerConfig
from live_transcript_cloud_worker.models import MediaType, StreamInfo
from live_transcript_cloud_worker.state import ChannelState
from live_transcript_cloud_worker.watcher import ChannelWatcher

YT_INFO = StreamInfo(url="https://www.youtube.com/watch?v=vid1", stream_id="vid1", is_live=True, start_time=int(time.time()))
YT_SHORT_INFO = StreamInfo(url="https://youtu.be/vid1", stream_id="vid1", is_live=True, start_time=int(time.time()))
TW_INFO = StreamInfo(url="https://www.twitch.tv/chan", stream_id="tw1", is_live=True, start_time=int(time.time()))


async def emit(chunk) -> None:
    pass


class FakeTimeline:
    """Stands in for TimelineCapture; records how the watcher builds and runs it."""

    instances: list[FakeTimeline] = []

    def __init__(self, config, streamer, state) -> None:
        self.args = (config, streamer, state)
        self.runs: list[tuple[StreamInfo, object, object]] = []
        FakeTimeline.instances.append(self)

    async def run(self, info, emit, should_stop) -> TimelineStats:
        self.runs.append((info, emit, should_stop))
        # A run that produced nothing at all: still no live-edge fallback.
        return TimelineStats()


class Calls:
    def __init__(self) -> None:
        self.live_edge: list[StreamInfo] = []

    async def capture_stream(self, config, streamer, state, info, emit, should_stop) -> None:
        self.live_edge.append(info)


@pytest.fixture
def calls(monkeypatch) -> Calls:
    recorder = Calls()
    monkeypatch.setattr(watcher_module, "capture_stream", recorder.capture_stream)
    return recorder


@pytest.fixture
def fake_timeline(monkeypatch):
    FakeTimeline.instances = []
    monkeypatch.setattr(watcher_module, "TimelineCapture", FakeTimeline)
    return FakeTimeline


def build_watcher(tmp_path, use_youtube_timeline: bool) -> ChannelWatcher:
    streamer = StreamerConfig(
        key="chan",
        urls=("https://www.youtube.com/@chan/live", "https://www.twitch.tv/chan"),
        media_type=MediaType.AUDIO,
        use_youtube_timeline=use_youtube_timeline,
    )
    config = make_config(tmp_path, streamers=(streamer,))
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    return ChannelWatcher(
        config,
        streamer,
        state,
        server=None,
        prober=None,
        transcriber=None,
        uploader=None,
        stop_event=asyncio.Event(),
    )


async def test_timeline_used_exclusively_for_youtube(tmp_path, calls, fake_timeline):
    watcher = build_watcher(tmp_path, use_youtube_timeline=True)
    (timeline,) = fake_timeline.instances
    assert timeline.args == (watcher.config, watcher.streamer, watcher.state)

    await watcher._capture(YT_INFO, emit)

    assert [run[0] for run in timeline.runs] == [YT_INFO]
    _, run_emit, should_stop = timeline.runs[0]
    assert run_emit is emit
    # The watcher's own stop check: a restart signal or shutdown ends the run.
    assert should_stop() is False
    watcher.restart_event.set()
    assert should_stop() is True
    # The run returned (here having produced nothing): capture is over for this
    # round, never continued on live-edge, so no line is ever approximate.
    assert calls.live_edge == []


async def test_timeline_instance_is_kept_across_captures(tmp_path, calls, fake_timeline):
    """One TimelineCapture per channel, so its yt-dlp pacing survives a run
    ending and the watcher capturing the same stream again."""
    watcher = build_watcher(tmp_path, use_youtube_timeline=True)

    await watcher._capture(YT_INFO, emit)
    await watcher._capture(YT_SHORT_INFO, emit)

    assert len(fake_timeline.instances) == 1
    assert [run[0] for run in fake_timeline.instances[0].runs] == [YT_INFO, YT_SHORT_INFO]
    assert calls.live_edge == []


async def test_timeline_flag_leaves_twitch_on_live_edge(tmp_path, calls, fake_timeline):
    watcher = build_watcher(tmp_path, use_youtube_timeline=True)

    await watcher._capture(TW_INFO, emit)

    assert calls.live_edge == [TW_INFO]
    assert fake_timeline.instances[0].runs == []


async def test_flag_off_uses_live_edge_for_youtube(tmp_path, calls, fake_timeline):
    watcher = build_watcher(tmp_path, use_youtube_timeline=False)

    await watcher._capture(YT_INFO, emit)
    await watcher._capture(TW_INFO, emit)

    assert calls.live_edge == [YT_INFO, TW_INFO]
    assert fake_timeline.instances[0].runs == []


async def test_watcher_builds_the_real_timeline_capture(tmp_path, calls, monkeypatch):
    """The real class: built once per watcher from (config, streamer, state),
    and its run() is what a timeline capture awaits."""
    watcher = build_watcher(tmp_path, use_youtube_timeline=True)
    assert isinstance(watcher._timeline, TimelineCapture)

    runs: list[StreamInfo] = []

    async def fake_run(info, emit, should_stop) -> TimelineStats:
        runs.append(info)
        return TimelineStats()

    monkeypatch.setattr(watcher._timeline, "run", fake_run)

    await watcher._capture(YT_INFO, emit)

    assert runs == [YT_INFO]
    assert calls.live_edge == []
