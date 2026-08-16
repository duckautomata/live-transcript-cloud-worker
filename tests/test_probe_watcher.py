"""Probe failure classification and watcher scheduling rules."""

from __future__ import annotations

import asyncio
import time

from conftest import make_config

from live_transcript_cloud_worker.models import Platform, ProbeResult, StreamInfo
from live_transcript_cloud_worker.state import ChannelState
from live_transcript_cloud_worker.watcher import ChannelWatcher, _UrlSchedule
from live_transcript_cloud_worker.ytdlp import (
    ProbeOutcome,
    _info_from_metadata,
    classify_probe_failure,
    platform_of,
    strip_trailing_timestamp,
)

TWITCH_OFFLINE = "ERROR: [twitch:stream] somechan: The channel is not currently live"
YT_OFFLINE = "ERROR: [youtube] abc: This channel is not currently live"
YT_SCHEDULED = "ERROR: [youtube] abc: This live event will begin in 3 hours."


def test_platform_of():
    assert platform_of("https://www.twitch.tv/x") is Platform.TWITCH
    assert platform_of("https://www.youtube.com/watch?v=1") is Platform.YOUTUBE
    assert platform_of("https://kick.com/x") is Platform.OTHER


def test_offline_confirmation_is_platform_generic():
    # yt-dlp raises the shared UserNotLive error for every platform,
    # Twitch included, confirmed offline must fire for all of them.
    for platform, stderr in (
        (Platform.TWITCH, TWITCH_OFFLINE),
        (Platform.YOUTUBE, YT_OFFLINE),
        (Platform.OTHER, "The channel is not currently live"),
    ):
        scheduled, confirmed = classify_probe_failure(stderr, platform)
        assert scheduled is None
        assert confirmed, platform


def test_scheduled_start_parsed_for_youtube_only():
    scheduled, confirmed = classify_probe_failure(YT_SCHEDULED, Platform.YOUTUBE)
    assert scheduled is not None
    assert time.time() + 2.5 * 3600 < scheduled < time.time() + 3.5 * 3600
    assert not confirmed
    # Twitch has no scheduled-start message; the parse is skipped.
    scheduled, confirmed = classify_probe_failure(YT_SCHEDULED, Platform.TWITCH)
    assert scheduled is None


def test_unrecognised_error_is_not_confirmed():
    scheduled, confirmed = classify_probe_failure("ERROR: something odd", Platform.YOUTUBE)
    assert scheduled is None
    assert not confirmed


# ------------------------------------------------------------------- titles


def test_strip_trailing_timestamp():
    # yt-dlp stamps live titles with the fetch time; it must not reach the server.
    assert strip_trailing_timestamp("the Wednesday stream 2026-08-12 17:38") == "the Wednesday stream"
    assert strip_trailing_timestamp("the Wednesday stream 2026-08-12 17:38:09") == "the Wednesday stream"
    assert strip_trailing_timestamp("Title - 2026-08-12T17:38") == "Title"
    assert strip_trailing_timestamp("Title (2026-08-12 17:38)") == "Title"
    assert strip_trailing_timestamp("Title [12/08/2026 5:30 PM]") == "Title"
    assert strip_trailing_timestamp("Stream Title 2023-01-01") == "Stream Title"
    assert strip_trailing_timestamp("Title 12:00") == "Title"


def test_strip_trailing_timestamp_keeps_the_broadcaster_s_own_text():
    assert strip_trailing_timestamp("Clean Title") == "Clean Title"
    assert strip_trailing_timestamp("2023-01-01 Stream Title") == "2023-01-01 Stream Title"
    assert strip_trailing_timestamp("Stream 12/12/2023 Title") == "Stream 12/12/2023 Title"
    # Only one stamp is removed, so a time the broadcaster wrote survives.
    assert strip_trailing_timestamp("karaoke 21:00 JST 2026-08-12 17:38") == "karaoke 21:00 JST"
    assert strip_trailing_timestamp("Best of (Part 1) 12:00") == "Best of (Part 1)"
    # A title that is nothing but a stamp is kept: empty is worse than noisy.
    assert strip_trailing_timestamp("2026-08-12 17:38") == "2026-08-12 17:38"


def test_probe_title_has_no_timestamp():
    info = _info_from_metadata(
        "https://www.youtube.com/watch?v=abc",
        {"id": "abc", "is_live": True, "title": "the Wednesday stream 2026-08-12 17:38", "release_timestamp": 1000},
    )
    assert info.title == "the Wednesday stream"

    twitch = _info_from_metadata(
        "https://www.twitch.tv/x",
        {"id": "123", "is_live": True, "display_id": "x", "description": "just chatting", "timestamp": 1000},
    )
    assert twitch.title == "x - just chatting"


# --------------------------------------------------------------- scheduling


def make_watcher(tmp_path, urls) -> ChannelWatcher:
    config = make_config(tmp_path)
    streamer = type(config.active_streamers[0])(key="chan", urls=tuple(urls))
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


def offline_outcome(confirmed: bool) -> ProbeOutcome:
    return ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url="u"), confirmed_offline=confirmed)


def test_confirmed_offline_backs_off_youtube_but_not_twitch(tmp_path):
    yt = "https://www.youtube.com/@x/live"
    tw = "https://www.twitch.tv/x"
    watcher = make_watcher(tmp_path, [yt, tw])
    polling = watcher.config.server.channel_polling

    schedule = _UrlSchedule()
    watcher._schedule_after_probe(schedule, offline_outcome(True), yt)
    assert schedule.next_check >= time.time() + polling.max_interval_seconds - 1

    schedule = _UrlSchedule()
    watcher._schedule_after_probe(schedule, offline_outcome(True), tw)
    # Twitch streams start unannounced: base cadence, never the long back-off.
    assert schedule.next_check <= time.time() + polling.interval_seconds + 11
    assert schedule.offline_count == 1


def test_unconfirmed_offline_uses_base_interval(tmp_path):
    yt = "https://www.youtube.com/@x/live"
    watcher = make_watcher(tmp_path, [yt])
    schedule = _UrlSchedule()
    watcher._schedule_after_probe(schedule, offline_outcome(False), yt)
    polling = watcher.config.server.channel_polling
    assert schedule.next_check <= time.time() + polling.interval_seconds + 11
    assert schedule.offline_count == 0


def test_upcoming_wakes_before_scheduled_start(tmp_path):
    yt = "https://www.youtube.com/@x/live"
    watcher = make_watcher(tmp_path, [yt])
    polling = watcher.config.server.channel_polling
    start = int(time.time()) + 3600
    outcome = ProbeOutcome(ProbeResult.UPCOMING, StreamInfo(url=yt), scheduled_start=start)
    schedule = _UrlSchedule()
    schedule.offline_count = 5
    watcher._schedule_after_probe(schedule, outcome, yt)
    assert schedule.offline_count == 0  # scheduled streams never count offline
    assert schedule.next_check <= start - polling.pre_scheduled_buffer_seconds + 1


# ----------------------------------------------------------------- restart


class RestartStubServer:
    def __init__(self, incoming=(), deactivate_ok=True):
        self.incoming = list(incoming)
        self.deactivated = []
        self.deleted = []
        self.acks = 0
        self.deactivate_ok = deactivate_ok

    async def deactivate(self, key, stream_id):
        self.deactivated.append(stream_id)
        return self.deactivate_ok

    async def ack_restart(self, key):
        self.acks += 1
        return True

    async def get_incoming(self, key):
        return list(self.incoming)

    async def delete_incoming(self, key, url):
        self.deleted.append(url)
        return True


def make_restart_watcher(tmp_path, incoming_mode: bool, server) -> ChannelWatcher:
    from live_transcript_cloud_worker.config import IncomingPollingConfig, ServerConfig

    config = make_config(
        tmp_path,
        server=ServerConfig(
            api_key="k",
            url="http://s.test",
            enabled=True,
            incoming_polling=IncomingPollingConfig(enabled=incoming_mode),
        ),
    )
    streamer = type(config.active_streamers[0])(key="chan", urls=("https://www.twitch.tv/x",))
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    return ChannelWatcher(
        config,
        streamer,
        state,
        server=server,
        prober=None,
        transcriber=None,
        uploader=None,
        stop_event=asyncio.Event(),
    )


async def test_restart_rebuilds_incoming_urls_but_keeps_state(tmp_path):
    from live_transcript_cloud_worker.models import Line, MediaType, Segment
    from live_transcript_cloud_worker.watcher import _UrlSchedule

    # Admin-stop cleared the server queue; the worker's local set still
    # holds the stopped stream's URL and must NOT keep re-probing it.
    server = RestartStubServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    watcher.state.start_new_stream("oldstream", "T", 1000, MediaType.AUDIO)
    watcher.state.append_line(Line(0, 1000, (Segment(1000, "hi"),), False))
    stale = _UrlSchedule()
    watcher._schedules["https://www.youtube.com/watch?v=oldstream"] = stale
    watcher.restart_event.set()

    await watcher._handle_restart()

    assert server.deactivated == ["oldstream"]
    assert server.acks == 1
    assert watcher._schedules == {}  # rebuilt from the (empty) server queue
    assert not watcher.restart_event.is_set()
    # Channel state is KEPT: a restart is also used to force a yt-dlp
    # reconnect, and the re-detected stream must resume at the next line,
    # never restart from 0 (which would wipe the server's transcript).
    assert watcher.state.stream_id == "oldstream"
    assert watcher.state.next_line_id == 1
    assert watcher.state.is_live is False  # deactivated until re-detected


async def test_dropping_incoming_url_deactivates_a_still_live_stream(tmp_path):
    from live_transcript_cloud_worker.models import MediaType

    # Activation succeeded but the stream's own deactivate never ran (e.g.
    # activate failed before capture, or the worker crashed mid-stream):
    # removing the URL is the last point at which the stream ID is known.
    server = RestartStubServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    watcher.state.start_new_stream("oldstream", "T", 1000, MediaType.AUDIO)
    url = "https://www.youtube.com/watch?v=oldstream"
    watcher._schedules[url] = _UrlSchedule()

    await watcher._drop_incoming_url(url)

    assert server.deactivated == ["oldstream"]
    assert server.deleted == [url]
    assert watcher.state.is_live is False
    assert url not in watcher._schedules


async def test_dropping_incoming_url_sends_nothing_when_not_live(tmp_path):
    # The common path: the stream ended and was already deactivated, so the
    # removal must not generate a redundant request.
    server = RestartStubServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    url = "https://www.youtube.com/watch?v=never-live"
    watcher._schedules[url] = _UrlSchedule()

    await watcher._drop_incoming_url(url)

    assert server.deactivated == []
    assert server.deleted == [url]


async def test_failed_deactivate_is_retried_on_the_next_removal(tmp_path):
    from live_transcript_cloud_worker.models import MediaType

    server = RestartStubServer(incoming=[], deactivate_ok=False)
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    watcher.state.start_new_stream("oldstream", "T", 1000, MediaType.AUDIO)

    await watcher._deactivate("oldstream")
    assert watcher.state.is_live is False  # not live locally...
    assert watcher._pending_deactivate == "oldstream"  # ...but unacknowledged

    server.deactivate_ok = True
    await watcher._drop_incoming_url("https://www.youtube.com/watch?v=oldstream")

    assert server.deactivated == ["oldstream", "oldstream"]
    assert watcher._pending_deactivate == ""


async def test_server_side_queue_removal_deactivates(tmp_path):
    from live_transcript_cloud_worker.models import MediaType

    # The operator removed the URL from /incoming: nothing will re-capture
    # this stream, so it must not stay live on the server.
    server = RestartStubServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    watcher.state.start_new_stream("oldstream", "T", 1000, MediaType.AUDIO)
    watcher._schedules["https://www.youtube.com/watch?v=oldstream"] = _UrlSchedule()

    await watcher._refresh_incoming()

    assert server.deactivated == ["oldstream"]
    assert server.deleted == []  # the server already dropped it
    assert watcher._schedules == {}


async def test_restart_static_mode_keeps_config_urls_and_reprobes(tmp_path):
    from live_transcript_cloud_worker.models import MediaType

    server = RestartStubServer()
    watcher = make_restart_watcher(tmp_path, incoming_mode=False, server=server)
    watcher.state.start_new_stream("oldstream", "T", 1000, MediaType.AUDIO)
    url = next(iter(watcher._schedules))
    watcher._schedules[url].next_check = time.time() + 9999
    watcher._schedules[url].offline_count = 2
    watcher.restart_event.set()

    await watcher._handle_restart()

    assert watcher.state.stream_id == "oldstream"  # state kept for resume
    assert list(watcher._schedules) == [url]  # config URLs stay
    assert watcher._schedules[url].next_check == 0.0  # re-probed immediately
    assert watcher._schedules[url].offline_count == 0
    assert not watcher.restart_event.is_set()
