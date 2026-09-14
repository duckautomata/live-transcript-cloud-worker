"""Probe failure classification and watcher scheduling rules."""

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import make_config

from live_transcript_cloud_worker.config import CaptureConfig, CookiesConfig, IncomingPollingConfig, ServerConfig
from live_transcript_cloud_worker.cookieauth import CookieAuth, CookieAuthTracker
from live_transcript_cloud_worker.models import Platform, ProbeResult, StreamInfo
from live_transcript_cloud_worker.state import ChannelState
from live_transcript_cloud_worker.watcher import (
    INCOMING_RETRY_BASE_SECONDS,
    POST_CAPTURE_RECHECK_SECONDS,
    ChannelWatcher,
    _UrlSchedule,
)
from live_transcript_cloud_worker.ytdlp import (
    ProbeOutcome,
    Prober,
    _info_from_metadata,
    classify_probe_failure,
    platform_of,
    strip_trailing_timestamp,
)

TWITCH_OFFLINE = "ERROR: [twitch:stream] somechan: The channel is not currently live"
YT_OFFLINE = "ERROR: [youtube] abc: This channel is not currently live"
YT_SCHEDULED = "ERROR: [youtube] abc: This live event will begin in 3 hours."
YT_LATE = "ERROR: [youtube] abc: This live event will begin in a few moments."
YT_PREMIERE = "ERROR: [youtube] abc: Premieres in 2 hours"
# The production incident: a dead jar (or a datacenter IP) hits the bot check
# on every probe. With --verbose yt-dlp appends a traceback after the ERROR line.
YT_BOT_CHECK = (
    "WARNING: [youtube] No title found in player responses; falling back to title from initial data. "
    "Other metadata may also be missing\n"
    "ERROR: [youtube] abc: Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies for the "
    "authentication. See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp  for how to "
    "manually pass cookies\n"
    '  File "/usr/local/bin/yt-dlp/yt_dlp/extractor/common.py", line 765, in extract\n'
    "    ie_result = self._real_extract(url)\n"
    '  File "/usr/local/bin/yt-dlp/yt_dlp/extractor/common.py", line 1272, in raise_no_formats\n'
    "    raise ExtractorError(msg, expected=expected, video_id=video_id)\n"
)


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
        verdict = classify_probe_failure(stderr, platform)
        assert not verdict.upcoming
        assert verdict.scheduled_start is None
        assert verdict.confirmed_offline, platform


def test_scheduled_start_parsed_for_youtube_only():
    verdict = classify_probe_failure(YT_SCHEDULED, Platform.YOUTUBE)
    assert verdict.upcoming
    assert verdict.scheduled_start is not None
    assert time.time() + 2.5 * 3600 < verdict.scheduled_start < time.time() + 3.5 * 3600
    assert not verdict.confirmed_offline
    # Twitch has no scheduled-start message; the parse is skipped.
    verdict = classify_probe_failure(YT_SCHEDULED, Platform.TWITCH)
    assert not verdict.upcoming
    assert verdict.scheduled_start is None


def test_upcoming_without_a_duration_is_still_upcoming():
    # A late streamer: YouTube drops the duration once the scheduled time has
    # passed. That is a positive "not started yet" verdict, not a failure the
    # watcher may count toward giving up on the URL.
    verdict = classify_probe_failure(YT_LATE, Platform.YOUTUBE)
    assert verdict.upcoming
    assert verdict.scheduled_start is None
    assert not verdict.confirmed_offline
    premiere = classify_probe_failure(YT_PREMIERE, Platform.YOUTUBE)
    assert premiere.upcoming
    assert premiere.scheduled_start is not None
    assert time.time() + 1.5 * 3600 < premiere.scheduled_start < time.time() + 2.5 * 3600


def test_unrecognised_error_is_not_confirmed():
    for stderr in ("ERROR: something odd", YT_BOT_CHECK):
        verdict = classify_probe_failure(stderr, Platform.YOUTUBE)
        assert not verdict.upcoming
        assert verdict.scheduled_start is None
        assert not verdict.confirmed_offline


WORKER_SIDE_STDERRS = (
    "ERROR: [youtube] abc: Unable to download API page: [Errno 111] Connection refused (caused by TransportError)",
    "ERROR: [youtube] abc: Unable to download webpage: <urlopen error [Errno -2] Name or service not known>",
    "ERROR: [youtube] abc: Unable to download API page: HTTP Error 429: Too Many Requests",
    "ERROR: [youtube] abc: This content isn't available, try again later. The current session has been rate-limited by YouTube for up to an hour.",
    "ERROR: [youtube] abc: Failed to extract any player response",
    "ERROR: [youtube] abc: All player responses are invalid. Your IP is likely being blocked by Youtube",
    "ERROR: [twitch:stream] somechan: Unable to download JSON metadata: HTTP Error 503: Service Unavailable",
)


def test_worker_side_failures_are_flagged():
    # YouTube unreachable, rate-limited, or the extractor broken: about this
    # worker, never about the URL, so the watcher must grade them ERROR.
    for stderr in WORKER_SIDE_STDERRS:
        platform = Platform.TWITCH if "twitch" in stderr else Platform.YOUTUBE
        verdict = classify_probe_failure(stderr, platform)
        assert verdict.worker_side, stderr
        assert not verdict.upcoming and not verdict.confirmed_offline
    for stderr in (YT_BOT_CHECK, "ERROR: [youtube] abc: Private video. Sign in if you've been granted access to this video"):
        assert not classify_probe_failure(stderr, Platform.YOUTUBE).worker_side
    # A handle that does not exist: yt-dlp retries the tab page with WARNING
    # lines that carry the generic "Unable to download webpage" prefix, then
    # fails with a per-URL 404. That is a verdict on the URL.
    missing_handle = (
        "WARNING: [youtube:tab] @nope/live: Unable to download webpage: HTTP Error 404: Not Found. Retrying (1/3)...\n"
        "WARNING: [youtube:tab] @nope/live: Unable to download webpage: HTTP Error 404: Not Found. Giving up after 3 retries\n"
        "WARNING: [youtube:tab] YouTube said: ERROR - Requested entity was not found.\n"
        "ERROR: [youtube:tab] @nope/live: Unable to download API page: HTTP Error 404: Not Found (caused by <HTTPError 404: Not Found>)\n"
    )
    assert not classify_probe_failure(missing_handle, Platform.YOUTUBE).worker_side
    # ...but retry WARNINGs never relabel the ERROR line either way.
    assert classify_probe_failure(
        "WARNING: something odd\nERROR: [youtube] abc: Unable to download API page: HTTP Error 503", Platform.YOUTUBE
    ).worker_side
    assert not classify_probe_failure("", Platform.YOUTUBE).worker_side
    # A confirmed offline that happens to mention a timeout elsewhere stays offline.
    assert classify_probe_failure(YT_OFFLINE + "\nWARNING: retry timed out", Platform.YOUTUBE).confirmed_offline


def test_probe_outcome_inconclusive_excludes_errors():
    unconfirmed = ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url="u"), confirmed_offline=False)
    confirmed = ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url="u"), confirmed_offline=True)
    error = ProbeOutcome(ProbeResult.ERROR, StreamInfo(url="u"), reason="probe timed out after 30s")
    assert unconfirmed.inconclusive
    assert not confirmed.inconclusive
    assert not error.inconclusive
    assert not ProbeOutcome(ProbeResult.LIVE, StreamInfo(url="u")).inconclusive


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


def test_terminal_status_is_confirmed_offline():
    # A finished YouTube stream resolves successfully (rc=0, is_live=false) and
    # reads as "was_live", never yt-dlp's "not currently live" failure. It must
    # still classify as confirmed offline, or an ended stream is re-probed at
    # the base cadence forever and never leaves the incoming queue.
    for status in ("was_live", "post_live", "not_live"):
        info = _info_from_metadata(
            "https://www.youtube.com/watch?v=abc",
            {"id": "abc", "is_live": False, "live_status": status, "title": "t"},
        )
        assert info.is_terminal_offline, status


def test_live_and_unknown_are_not_terminal_offline():
    live = _info_from_metadata(
        "https://www.youtube.com/watch?v=abc",
        {"id": "abc", "is_live": True, "live_status": "is_live", "title": "t"},
    )
    assert not live.is_terminal_offline
    # An empty/unknown status is transient (yt-dlp emits it between states):
    # acting on it could drop a URL that is about to go live.
    unknown = _info_from_metadata(
        "https://www.youtube.com/watch?v=abc",
        {"id": "abc", "is_live": False, "title": "t"},
    )
    assert unknown.live_status == "unknown"
    assert not unknown.is_terminal_offline


async def test_probe_reports_ended_stream_as_confirmed_offline(tmp_path):
    # End to end through the real Prober subprocess: a finished stream is a
    # *successful* probe (rc=0) whose metadata says was_live. It must come back
    # confirmed offline so the watcher backs it off and drops it from the queue
    # instead of re-probing every minute forever.
    fake = tmp_path / "fake-yt-dlp"
    fake.write_text('#!/bin/bash\necho \'{"id": "abc", "live_status": "was_live", "title": "t"}\'\n')
    fake.chmod(0o755)
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))

    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")

    assert outcome.result is ProbeResult.OFFLINE
    assert outcome.confirmed_offline
    assert outcome.reason == "live_status=was_live"


def _fake_ytdlp(tmp_path, script: str):
    fake = tmp_path / "fake-yt-dlp"
    fake.write_text("#!/bin/bash\n" + script)
    fake.chmod(0o755)
    return fake


async def test_probe_reason_is_the_error_line_not_the_traceback(tmp_path):
    # --verbose makes yt-dlp print a traceback after its ERROR line, so the
    # last line of stderr is "raise ExtractorError(...)"; the operator-facing
    # reason must be the ERROR line itself.
    stderr = YT_BOT_CHECK.replace("'", "'\\''")
    fake = _fake_ytdlp(tmp_path, f"printf '%s' '{stderr}' >&2\nexit 1\n")
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))

    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")

    assert outcome.result is ProbeResult.OFFLINE
    assert not outcome.confirmed_offline
    assert outcome.inconclusive
    assert outcome.reason.startswith("ERROR: [youtube] abc: Sign in to confirm you're not a bot")
    assert "raise ExtractorError" not in outcome.reason


async def test_probe_timeout_is_an_error_with_a_reason(tmp_path):
    fake = _fake_ytdlp(tmp_path, "sleep 5\n")
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake), probe_timeout_seconds=0.2))

    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")

    assert outcome.result is ProbeResult.ERROR
    assert not outcome.inconclusive
    assert outcome.reason == "probe timed out after 0.2s"


async def test_probe_network_failure_is_an_error_not_a_verdict(tmp_path, caplog):
    fake = _fake_ytdlp(tmp_path, f"echo '{WORKER_SIDE_STDERRS[0]}' >&2\nexit 1\n")
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))

    with caplog.at_level("WARNING", logger="live_transcript_cloud_worker.ytdlp"):
        outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")

    assert outcome.result is ProbeResult.ERROR
    assert not outcome.inconclusive
    assert "Connection refused" in outcome.reason
    assert any("worker's side" in r.getMessage() for r in caplog.records)


async def test_match_filter_rejection_is_inconclusive_not_an_error(tmp_path):
    # --match-filter rejects a members-only entry: yt-dlp exits 0 with no
    # JSON at all. That is a verdict on the URL (it can never resolve for
    # us), so it must reach the give-up rather than be re-probed forever.
    fake = _fake_ytdlp(
        tmp_path,
        "echo '[download] t does not pass filter (availability!=?subscriber_only), skipping ..' >&2\nexit 0\n",
    )
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))

    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")

    assert outcome.result is ProbeResult.OFFLINE
    assert outcome.inconclusive
    assert not outcome.confirmed_offline
    assert "does not pass filter" in outcome.reason


async def test_signal_killed_or_silent_ytdlp_is_an_error(tmp_path):
    # The OOM killer, or a wrapper that died without a word: no verdict.
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(_fake_ytdlp(tmp_path, "kill -9 $$\n"))))
    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")
    assert outcome.result is ProbeResult.ERROR
    assert outcome.reason == "yt-dlp killed by signal 9"

    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(_fake_ytdlp(tmp_path, "exit 1\n"))))
    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")
    assert outcome.result is ProbeResult.ERROR
    assert outcome.reason == "yt-dlp printed nothing"

    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(_fake_ytdlp(tmp_path, "exit 0\n"))))
    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")
    assert outcome.result is ProbeResult.ERROR  # rc 0, nothing on either stream: a wrong binary, not a verdict


async def test_match_filter_reason_survives_a_preceding_warning(tmp_path):
    fake = _fake_ytdlp(
        tmp_path,
        "echo 'WARNING: [youtube] abc: nsig extraction failed: Some formats may be missing' >&2\n"
        "echo '[download] Members stream does not pass filter (availability!=?subscriber_only), skipping ..' >&2\nexit 0\n",
    )
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))
    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")
    assert outcome.inconclusive
    assert outcome.reason == "no metadata: [download] Members stream does not pass filter (availability!=?subscriber_only), skipping .."


async def test_garbage_output_is_still_an_error(tmp_path):
    fake = _fake_ytdlp(tmp_path, "echo 'not json'\nexit 0\n")
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))

    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")

    assert outcome.result is ProbeResult.ERROR
    assert outcome.reason == "unparseable yt-dlp output"


async def test_probe_late_stream_is_upcoming_without_a_start(tmp_path):
    fake = _fake_ytdlp(tmp_path, f"echo '{YT_LATE}' >&2\nexit 1\n")
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))

    outcome = await Prober(config).probe("https://www.youtube.com/watch?v=abc")

    assert outcome.result is ProbeResult.UPCOMING
    assert outcome.scheduled_start is None


# ------------------------------------------------------- jar replacement


def _jar_text(login_info: str = "AFmmF2sw", sapisid: str = "abc123", visitor: str = "v1", ysc: str = "y1") -> str:
    """A Netscape jar the way yt-dlp writes it: HttpOnly auth cookies carry the
    "#HttpOnly_" prefix, and the per-request cookies churn on every run."""
    return (
        "# Netscape HTTP Cookie File\n"
        "# This file is generated by yt-dlp.  Do not edit.\n\n"
        f"#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t1800000000\tLOGIN_INFO\t{login_info}\n"
        f".youtube.com\tTRUE\t/\tTRUE\t1800000000\tSAPISID\t{sapisid}\n"
        f".youtube.com\tTRUE\t/\tTRUE\t1800000000\t__Secure-3PAPISID\t{sapisid}\n"
        f".youtube.com\tTRUE\t/\tTRUE\t1800000000\tVISITOR_INFO1_LIVE\t{visitor}\n"
        f".youtube.com\tTRUE\t/\tTRUE\t0\tYSC\t{ysc}\n"
        f".youtube.com\tTRUE\t/\tTRUE\t1800000000\t__Secure-1PSIDTS\tsidts-{visitor}\n"
    )


def _cookie_config(tmp_path, **kwargs):
    (tmp_path / "cookies.txt").write_text(_jar_text())
    server = ServerConfig(api_key="k", url="http://s.test", enabled=True, cookies=CookiesConfig(enabled=True))
    return make_config(tmp_path, server=server, **kwargs)


# A fake yt-dlp that rewrites the jar the way the real one does on exit: same
# auth cookies, fresh per-request ones, new mtime and size.
_REWRITE_JAR = (
    'while [ $# -gt 0 ]; do if [ "$1" = "--cookies" ]; then JAR="$2"; fi; shift; done\n'
    'printf "# Netscape HTTP Cookie File\\n# This file is generated by yt-dlp.  Do not edit.\\n\\n" > "$JAR"\n'
    'printf "#HttpOnly_.youtube.com\\tTRUE\\t/\\tTRUE\\t1800000000\\tLOGIN_INFO\\tAFmmF2sw\\n" >> "$JAR"\n'
    'printf ".youtube.com\\tTRUE\\t/\\tTRUE\\t1800000000\\tSAPISID\\tabc123\\n" >> "$JAR"\n'
    'printf ".youtube.com\\tTRUE\\t/\\tTRUE\\t1800000000\\t__Secure-3PAPISID\\tabc123\\n" >> "$JAR"\n'
    'printf ".youtube.com\\tTRUE\\t/\\tTRUE\\t1800000000\\tVISITOR_INFO1_LIVE\\tv$$\\n" >> "$JAR"\n'
    'printf ".youtube.com\\tTRUE\\t/\\tTRUE\\t0\\tYSC\\ty$$$$\\n" >> "$JAR"\n'
    'printf ".youtube.com\\tTRUE\\t/\\tTRUE\\t1800000000\\t__Secure-1PSIDTS\\tsidts-$$\\n" >> "$JAR"\n'
)
JSON_LINE = 'echo \'{"id": "abc", "live_status": "was_live", "title": "t"}\'\n'


async def test_poll_jar_ignores_the_probes_own_rewrite_but_sees_the_operators(tmp_path):
    # yt-dlp rewrites cookies.txt on every normal exit (new mtime, new size,
    # churned per-request cookies); that must not look like the operator
    # dropping in a fresh jar, which changes the auth cookies.
    fake = _fake_ytdlp(tmp_path, _REWRITE_JAR + JSON_LINE)
    config = _cookie_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))
    prober = Prober(config)
    assert prober.poll_jar() == 0
    before = (tmp_path / "cookies.txt").read_text()

    await prober.probe("https://www.youtube.com/watch?v=abc")
    assert (tmp_path / "cookies.txt").read_text() != before  # really rewritten
    assert prober.poll_jar() == 0  # our own write

    (tmp_path / "cookies.txt").write_text(_jar_text(login_info="FRESH", sapisid="fresh456"))
    assert prober.poll_jar() == 1  # the operator's write
    assert prober.poll_jar() == 1  # reported once

    (tmp_path / "cookies.txt").write_text(_jar_text(login_info="FRESH", sapisid="fresh456", visitor="v9"))
    assert prober.poll_jar() == 1  # per-request cookies alone are not a new jar


async def test_poll_jar_during_a_probe_never_sees_a_torn_jar(tmp_path):
    # yt-dlp's rewrite truncates the file and then writes it, shortly BEFORE
    # it exits; a tick from another channel's watcher landing in that window
    # would read a jar with no auth cookies. The in-flight guard makes the
    # window invisible.
    fake = _fake_ytdlp(
        tmp_path,
        'while [ $# -gt 0 ]; do if [ "$1" = "--cookies" ]; then JAR="$2"; fi; shift; done\n'
        'cp "$JAR" "$JAR.full"\n'
        ': > "$JAR"\n'
        "sleep 0.3\n"
        'cat "$JAR.full" > "$JAR"\n' + JSON_LINE,
    )
    config = _cookie_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(fake)))
    prober = Prober(config)
    generations = []

    async def poller():
        for _ in range(60):
            generations.append(prober.poll_jar())
            await asyncio.sleep(0.01)

    await asyncio.gather(prober.probe("https://www.youtube.com/watch?v=abc"), poller())

    assert set(generations) == {0}
    assert prober.poll_jar() == 0
    (tmp_path / "cookies.txt").write_text(_jar_text(login_info="FRESH"))
    assert prober.poll_jar() == 1  # the operator's write is still seen afterwards


async def test_capture_exit_does_not_wake_parked_urls(tmp_path, monkeypatch):
    # The download's yt-dlp rewrites the shared jar on exit too, with nobody
    # holding the in-flight guard; only the auth cookies count, so it is
    # invisible even to a tick that lands mid-capture.
    from live_transcript_cloud_worker import watcher as watcher_module

    config = _cookie_config(tmp_path)
    prober = Prober(config)
    streamer = type(config.active_streamers[0])(key="chan", urls=(YT_URL,))
    watcher = ChannelWatcher(
        config,
        streamer,
        ChannelState("chan", tmp_path / "tmp" / "chan"),
        server=None,
        prober=prober,
        transcriber=None,
        uploader=None,
        stop_event=asyncio.Event(),
    )

    async def fake_capture_stream(config, streamer, state, info, emit, should_stop):
        (tmp_path / "cookies.txt").write_text(_jar_text(visitor="v-from-download", ysc="y-from-download"))
        assert prober.poll_jar() == 0

    monkeypatch.setattr(watcher_module, "capture_stream", fake_capture_stream)
    await watcher._capture(StreamInfo(url=YT_URL, stream_id="s1"), lambda chunk: None)

    assert prober.poll_jar() == 0


def test_rotated_jar_reads_as_a_change(tmp_path):
    # YouTube invalidated the session: yt-dlp writes the jar back without
    # LOGIN_INFO. That is a real change in the jar's identity.
    prober = Prober(_cookie_config(tmp_path))
    text = (tmp_path / "cookies.txt").read_text()
    (tmp_path / "cookies.txt").write_text("\n".join(line for line in text.splitlines() if "LOGIN_INFO" not in line) + "\n")
    assert prober.poll_jar() == 1


def test_poll_jar_is_inert_without_cookies(tmp_path):
    prober = Prober(make_config(tmp_path))
    assert prober.poll_jar() == 0
    (tmp_path / "cookies.txt").write_text("x")
    assert prober.poll_jar() == 0


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


def test_unconfirmed_offline_uses_base_interval_and_counts_inconclusive(tmp_path):
    yt = "https://www.youtube.com/@x/live"
    watcher = make_watcher(tmp_path, [yt])
    schedule = _UrlSchedule()
    schedule.offline_count = 2
    watcher._schedule_after_probe(schedule, offline_outcome(False), yt)
    polling = watcher.config.server.channel_polling
    assert schedule.next_check <= time.time() + polling.interval_seconds + 11
    assert schedule.offline_count == 2  # frozen, not reset: a real streak resumes later
    assert schedule.inconclusive_count == 1


def test_error_outcome_neither_counts_nor_resets(tmp_path):
    # A timeout or a broken yt-dlp says nothing about the URL.
    yt = "https://www.youtube.com/@x/live"
    watcher = make_watcher(tmp_path, [yt])
    schedule = _UrlSchedule()
    schedule.offline_count = 2
    schedule.inconclusive_count = 4
    error = ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=yt), reason="probe timed out after 30s")
    watcher._schedule_after_probe(schedule, error, yt)
    assert schedule.offline_count == 2
    assert schedule.inconclusive_count == 4
    assert schedule.next_check <= time.time() + watcher.config.server.channel_polling.interval_seconds + 11


def test_trusted_confirmed_offline_resets_inconclusive(tmp_path):
    yt = "https://www.youtube.com/@x/live"
    watcher = make_watcher(tmp_path, [yt])
    schedule = _UrlSchedule()
    schedule.inconclusive_count = 4
    schedule.parked = True
    watcher._schedule_after_probe(schedule, offline_outcome(True), yt)
    assert schedule.offline_count == 1
    assert schedule.inconclusive_count == 0
    assert not schedule.parked


def test_upcoming_wakes_before_scheduled_start(tmp_path):
    yt = "https://www.youtube.com/@x/live"
    watcher = make_watcher(tmp_path, [yt])
    polling = watcher.config.server.channel_polling
    start = int(time.time()) + 3600
    outcome = ProbeOutcome(ProbeResult.UPCOMING, StreamInfo(url=yt), scheduled_start=start)
    schedule = _UrlSchedule()
    schedule.offline_count = 5
    schedule.inconclusive_count = 3
    watcher._schedule_after_probe(schedule, outcome, yt)
    assert schedule.offline_count == 0  # scheduled streams never count offline
    assert schedule.inconclusive_count == 0
    assert schedule.next_check <= start - polling.pre_scheduled_buffer_seconds + 1


def test_upcoming_without_start_stays_on_base_cadence_and_resets(tmp_path):
    # "will begin in a few moments": the streamer is late, not missing.
    yt = "https://www.youtube.com/watch?v=abc"
    watcher = make_watcher(tmp_path, [yt])
    polling = watcher.config.server.channel_polling
    schedule = _UrlSchedule()
    schedule.inconclusive_count = 9
    watcher._schedule_after_probe(schedule, ProbeOutcome(ProbeResult.UPCOMING, StreamInfo(url=yt)), yt)
    assert schedule.inconclusive_count == 0
    assert time.time() + polling.interval_seconds - 6 <= schedule.next_check <= time.time() + polling.interval_seconds + 11


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


def make_restart_watcher(
    tmp_path,
    incoming_mode: bool,
    server,
    *,
    prober=None,
    inconclusive_threshold: int = 10,
    inconclusive_min_span: float = 0.0,
    offline_threshold: int = 2,
    incoming_interval: float = 30.0,
    id_blacklist=(),
    stop_event: asyncio.Event | None = None,
) -> ChannelWatcher:
    config = make_config(
        tmp_path,
        server=ServerConfig(
            api_key="k",
            url="http://s.test",
            enabled=True,
            incoming_polling=IncomingPollingConfig(
                enabled=incoming_mode,
                interval_seconds=incoming_interval,
                offline_delete_threshold=offline_threshold,
                inconclusive_delete_threshold=inconclusive_threshold,
                inconclusive_min_span_seconds=inconclusive_min_span,
            ),
        ),
        id_blacklist=tuple(id_blacklist),
    )
    streamer = type(config.active_streamers[0])(key="chan", urls=("https://www.twitch.tv/x",))
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    return ChannelWatcher(
        config,
        streamer,
        state,
        server=server,
        prober=prober,
        transcriber=None,
        uploader=None,
        stop_event=stop_event or asyncio.Event(),
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

    assert await watcher._drop_incoming_url(url, "test")

    assert server.deactivated == ["oldstream"]
    assert server.deleted == [url]
    assert watcher.state.is_live is False
    assert url not in watcher._schedules


async def test_failed_queue_delete_keeps_the_url_scheduled(tmp_path):
    # The DELETE never landed: popping the schedule anyway would let the next
    # queue refresh re-add the URL with a clean slate and restart the streak.
    class NoDeleteServer(RestartStubServer):
        async def delete_incoming(self, key, url):
            self.deleted.append(url)
            return False

    server = NoDeleteServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    url = "https://www.youtube.com/watch?v=stuck"
    schedule = _UrlSchedule()
    schedule.inconclusive_count = 10
    watcher._schedules[url] = schedule

    assert not await watcher._drop_incoming_url(url, "test")

    assert server.deleted == [url]
    assert watcher._schedules[url] is schedule
    assert schedule.inconclusive_count == 10


async def test_dropping_incoming_url_sends_nothing_when_not_live(tmp_path):
    # The common path: the stream ended and was already deactivated, so the
    # removal must not generate a redundant request.
    server = RestartStubServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    url = "https://www.youtube.com/watch?v=never-live"
    watcher._schedules[url] = _UrlSchedule()

    await watcher._drop_incoming_url(url, "test")

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
    await watcher._drop_incoming_url("https://www.youtube.com/watch?v=oldstream", "test")

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
    watcher._schedules[url].inconclusive_count = 4
    watcher._schedules[url].parked = True
    watcher.restart_event.set()

    await watcher._handle_restart()

    assert watcher.state.stream_id == "oldstream"  # state kept for resume
    assert list(watcher._schedules) == [url]  # config URLs stay
    assert watcher._schedules[url].next_check == 0.0  # re-probed immediately
    assert watcher._schedules[url].offline_count == 0
    assert watcher._schedules[url].inconclusive_count == 0
    assert not watcher._schedules[url].parked
    assert not watcher.restart_event.is_set()


# ------------------------------------------- transient /incoming failures


class UnreachableIncomingServer(RestartStubServer):
    """get_incoming returns None: the server never answered."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.incoming_calls = 0

    async def get_incoming(self, key):
        self.incoming_calls += 1
        return None


async def test_unanswered_incoming_poll_keeps_known_urls(tmp_path):
    from live_transcript_cloud_worker.models import MediaType

    # A few-second server restart must not look like "the operator emptied
    # the queue": dropping the URLs here stops the channel being probed at
    # all, and deactivates a stream that is still running.
    server = UnreachableIncomingServer()
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    watcher.state.start_new_stream("livestream", "T", 1000, MediaType.AUDIO)
    url = "https://www.youtube.com/watch?v=livestream"
    watcher._schedules[url] = _UrlSchedule()

    await watcher._refresh_incoming()

    assert list(watcher._schedules) == [url]
    assert server.deactivated == []


async def test_unanswered_incoming_poll_retries_before_the_fallback_window(tmp_path):
    server = UnreachableIncomingServer()
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)

    await watcher._refresh_incoming()

    # The retry must land on the short backoff, not the (much longer)
    # fallback interval, and must not be due instantly either -- that would
    # re-poll on every 1 s tick for the whole outage.
    assert not watcher._incoming_fallback_elapsed()
    delay = watcher._last_incoming_refresh + watcher._incoming_interval() - time.time()
    assert 0 < delay <= INCOMING_RETRY_BASE_SECONDS + 1


async def test_repeated_incoming_failures_back_off_and_then_reset(tmp_path):
    server = UnreachableIncomingServer()
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)

    for _ in range(3):
        await watcher._refresh_incoming()
    assert watcher._incoming_failures == 3

    # A successful poll clears the backoff and applies the queue normally.
    watcher.server = RestartStubServer(incoming=["https://www.youtube.com/watch?v=fresh"])
    await watcher._refresh_incoming()

    assert watcher._incoming_failures == 0
    assert list(watcher._schedules) == ["https://www.youtube.com/watch?v=fresh"]


async def test_restart_during_outage_keeps_urls(tmp_path):
    # An admin-stop that lands during a redeploy used to clear the URL set
    # and then fail to refill it, silently disabling the channel.
    server = UnreachableIncomingServer()
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server)
    url = "https://www.youtube.com/watch?v=queued"
    watcher._schedules[url] = _UrlSchedule()
    watcher.restart_event.set()

    await watcher._handle_restart()

    assert list(watcher._schedules) == [url]
    assert not watcher.restart_event.is_set()


# ----------------------------------- verdict grades and the incoming queue


class ScriptedProber:
    """A prober that replays scripted outcomes (the last one repeats) and
    records every probe. ``stop_after`` probes sets the watcher's stop event
    so run() tests end deterministically."""

    def __init__(self, outcomes, cookies=None, stop_event=None, stop_after=None):
        self.outcomes = list(outcomes)
        self.cookies = cookies or CookieAuthTracker()
        self.calls: list[str] = []
        self.jar_generation = 0
        self._stop_event = stop_event
        self._stop_after = stop_after

    async def probe(self, url, state=None):
        self.calls.append(url)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if self._stop_event is not None and self._stop_after is not None and len(self.calls) >= self._stop_after:
            self._stop_event.set()
        return outcome

    def poll_jar(self):
        return self.jar_generation


def _tracker(state: CookieAuth) -> CookieAuthTracker:
    tracker = CookieAuthTracker()
    # ABSENT is debounced over CONSECUTIVE_TO_ALARM probes; OK and ROTATED
    # flip on the first observation, so repeating is harmless.
    for _ in range(tracker.threshold):
        tracker.observe(state)
    assert tracker.state is state
    return tracker


def bot_check(url: str) -> ProbeOutcome:
    return ProbeOutcome(
        ProbeResult.OFFLINE,
        StreamInfo(url=url),
        confirmed_offline=False,
        reason="ERROR: [youtube] abc: Sign in to confirm you're not a bot",
    )


YT_URL = "https://www.youtube.com/watch?v=abc"


async def test_degraded_cookies_park_the_url_instead_of_counting_offline(tmp_path):
    # Signed out, a /channel/<id>/live URL we cannot see raises yt-dlp's
    # UserNotLive -- the same "not currently live" text a genuine offline
    # produces. It must not advance the offline-delete streak; it counts as
    # inconclusive and the URL moves to the slow degraded cadence.
    server = RestartStubServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=ScriptedProber([], _tracker(CookieAuth.ROTATED)))
    degraded = watcher.config.server.cookies.degraded_probe_interval_seconds

    url = "https://www.youtube.com/channel/UCabc/live"
    schedule = _UrlSchedule()
    schedule.offline_count = 1
    outcome = ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=url), confirmed_offline=True)

    for _ in range(20):
        watcher._schedule_after_probe(schedule, outcome, url)

    assert schedule.offline_count == 1  # frozen where it was
    assert schedule.inconclusive_count == 20
    assert schedule.parked
    assert time.time() + degraded - 1 <= schedule.next_check <= time.time() + degraded + 11


async def test_degraded_error_probe_is_parked_but_not_counted(tmp_path):
    watcher = make_restart_watcher(
        tmp_path, incoming_mode=True, server=RestartStubServer(), prober=ScriptedProber([], _tracker(CookieAuth.ABSENT))
    )
    schedule = _UrlSchedule()
    error = ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=YT_URL), reason="probe timed out after 30s")
    watcher._schedule_after_probe(schedule, error, YT_URL)
    assert schedule.inconclusive_count == 0
    assert schedule.parked


async def test_healthy_cookies_still_count_offline_normally(tmp_path):
    server = RestartStubServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=ScriptedProber([], _tracker(CookieAuth.OK)))

    url = "https://www.youtube.com/channel/UCabc/live"
    schedule = _UrlSchedule()
    outcome = ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=url), confirmed_offline=True)

    for _ in range(3):
        watcher._schedule_after_probe(schedule, outcome, url)

    assert schedule.offline_count == 3
    assert schedule.inconclusive_count == 0
    assert not schedule.parked


async def test_twitch_is_unaffected_by_dead_youtube_cookies(tmp_path):
    # Twitch never gets cookies, so its offline verdicts stay authoritative.
    server = RestartStubServer(incoming=[])
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=ScriptedProber([], _tracker(CookieAuth.ROTATED)))

    url = "https://www.twitch.tv/somechan"
    schedule = _UrlSchedule()
    outcome = ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=url), confirmed_offline=True)

    for _ in range(3):
        watcher._schedule_after_probe(schedule, outcome, url)

    assert schedule.offline_count == 3
    assert not schedule.parked


async def test_bot_checked_url_is_given_up_on_at_the_inconclusive_threshold(tmp_path, caplog):
    # The production incident: the jar died on the URL's first probe and every
    # probe after that was "Sign in to confirm you're not a bot". Nothing about
    # the stream can be learned, so after the threshold the queue entry goes,
    # with a log line that says why and that the stream may still be live.
    server = RestartStubServer(incoming=[YT_URL])
    prober = ScriptedProber([bot_check(YT_URL)], _tracker(CookieAuth.ABSENT))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, inconclusive_threshold=3)
    watcher._schedules[YT_URL] = _UrlSchedule()

    with caplog.at_level("INFO", logger="live_transcript_cloud_worker.watcher"):
        for _ in range(2):
            await watcher._check_url(YT_URL)
        assert server.deleted == []
        assert watcher._schedules[YT_URL].inconclusive_count == 2
        await watcher._check_url(YT_URL)

    assert server.deleted == [YT_URL]
    assert YT_URL not in watcher._schedules
    assert prober.calls == [YT_URL] * 3
    giving_up = [r for r in caplog.records if "giving up" in r.getMessage()]
    assert len(giving_up) == 1
    assert giving_up[0].levelname == "WARNING"
    assert "Sign in to confirm" in giving_up[0].getMessage()
    assert giving_up[0].cookie_state == "absent"
    assert giving_up[0].inconclusive_count == 3
    parked = [r for r in caplog.records if "parking" in r.getMessage()]
    assert len(parked) == 1  # logged once, not on every probe


async def test_give_up_also_needs_the_minimum_span(tmp_path):
    # Ten inconclusive probes in ten minutes might be a worker-wide hiccup
    # (a datacenter bot check with a healthy jar); the streak must also have
    # lasted inconclusive_min_span_seconds before the queue is touched.
    server = RestartStubServer(incoming=[YT_URL])
    prober = ScriptedProber([bot_check(YT_URL)], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(
        tmp_path, incoming_mode=True, server=server, prober=prober, inconclusive_threshold=3, inconclusive_min_span=3600
    )
    watcher._schedules[YT_URL] = _UrlSchedule()

    for _ in range(6):
        await watcher._check_url(YT_URL)
    assert server.deleted == []
    assert watcher._schedules[YT_URL].inconclusive_count == 6

    watcher._schedules[YT_URL].inconclusive_since = time.time() - 3601
    await watcher._check_url(YT_URL)
    assert server.deleted == [YT_URL]


async def test_give_up_never_fires_on_an_error_probe(tmp_path, caplog):
    # Count and span satisfied, but the probe that runs next is a timeout:
    # the removal must wait for a probe that is actually a verdict.
    server = RestartStubServer(incoming=[YT_URL])
    timeout = ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=YT_URL), reason="probe timed out after 30s")
    prober = ScriptedProber([timeout], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, inconclusive_threshold=3)
    schedule = _UrlSchedule()
    schedule.inconclusive_count = 3
    schedule.inconclusive_since = time.time() - 5000
    watcher._schedules[YT_URL] = schedule

    with caplog.at_level("WARNING", logger="live_transcript_cloud_worker.watcher"):
        await watcher._check_url(YT_URL)
    assert server.deleted == []
    assert not any("giving up" in r.getMessage() for r in caplog.records)

    prober.outcomes = [bot_check(YT_URL)]
    await watcher._check_url(YT_URL)
    assert server.deleted == [YT_URL]


async def test_wake_triggered_probe_with_a_healthy_jar_does_not_count_either(tmp_path):
    server = RestartStubServer(incoming=[YT_URL])
    tracker = _tracker(CookieAuth.ABSENT)
    prober = ScriptedProber([bot_check(YT_URL)], tracker)
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober)
    schedule = _UrlSchedule()
    watcher._schedules[YT_URL] = schedule

    await watcher._check_url(YT_URL)
    assert schedule.parked and schedule.inconclusive_count == 1

    prober.jar_generation += 1
    watcher._wake_parked_if_jar_replaced()
    tracker.observe(CookieAuth.OK)  # the new jar authenticates, the datacenter IP is still bot-checked
    await watcher._check_url(YT_URL)

    assert schedule.inconclusive_count == 1  # the wake's probe never counts
    assert not schedule.parked and not schedule.woken
    await watcher._check_url(YT_URL)
    assert schedule.inconclusive_count == 2  # cadence probes do


async def test_wake_triggered_probe_does_not_count_and_reparks_loudly(tmp_path, caplog):
    server = RestartStubServer(incoming=[YT_URL])
    prober = ScriptedProber([bot_check(YT_URL)], _tracker(CookieAuth.ABSENT))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober)
    schedule = _UrlSchedule()
    watcher._schedules[YT_URL] = schedule

    with caplog.at_level("WARNING", logger="live_transcript_cloud_worker.watcher"):
        await watcher._check_url(YT_URL)
        assert schedule.parked and schedule.inconclusive_count == 1

        prober.jar_generation += 1  # the operator overwrote cookies.txt
        watcher._wake_parked_if_jar_replaced()
        assert schedule.next_check == 0.0 and schedule.woken and not schedule.parked

        await watcher._check_url(YT_URL)  # the new jar is dead too
        assert schedule.inconclusive_count == 1  # a wake costs one probe, never a count
        assert schedule.parked and not schedule.woken

        await watcher._check_url(YT_URL)  # back on the cadence: counts again
        assert schedule.inconclusive_count == 2

    parked = [r for r in caplog.records if "parking" in r.getMessage()]
    assert len(parked) == 2  # the re-park after the swap is logged again


async def test_recovered_then_dead_again_logs_parking_twice(tmp_path, caplog):
    server = RestartStubServer(incoming=[YT_URL])
    tracker = _tracker(CookieAuth.ABSENT)
    prober = ScriptedProber([bot_check(YT_URL)], tracker)
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober)
    schedule = _UrlSchedule()
    watcher._schedules[YT_URL] = schedule
    base = watcher.config.server.channel_polling.interval_seconds

    with caplog.at_level("WARNING", logger="live_transcript_cloud_worker.watcher"):
        await watcher._check_url(YT_URL)
        assert schedule.parked

        tracker.observe(CookieAuth.OK)  # jar healthy, but the datacenter IP is still bot-checked
        await watcher._check_url(YT_URL)
        assert not schedule.parked
        assert schedule.next_check <= time.time() + base + 11
        assert schedule.inconclusive_count == 2

        for _ in range(tracker.threshold):
            tracker.observe(CookieAuth.ABSENT)
        await watcher._check_url(YT_URL)
        assert schedule.parked

    assert len([r for r in caplog.records if "parking" in r.getMessage()]) == 2


async def test_worker_side_probe_failures_never_give_up(tmp_path):
    server = RestartStubServer(incoming=[YT_URL])
    outage = ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=YT_URL), reason=WORKER_SIDE_STDERRS[0])
    prober = ScriptedProber([outage], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, inconclusive_threshold=2)
    watcher._schedules[YT_URL] = _UrlSchedule()

    for _ in range(30):
        await watcher._check_url(YT_URL)

    assert server.deleted == []
    assert watcher._schedules[YT_URL].inconclusive_count == 0


async def test_inconclusive_threshold_zero_never_gives_up(tmp_path):
    server = RestartStubServer(incoming=[YT_URL])
    prober = ScriptedProber([bot_check(YT_URL)], _tracker(CookieAuth.ABSENT))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, inconclusive_threshold=0)
    watcher._schedules[YT_URL] = _UrlSchedule()

    for _ in range(30):
        await watcher._check_url(YT_URL)

    assert server.deleted == []
    assert watcher._schedules[YT_URL].inconclusive_count == 30


def confirmed_offline(url: str) -> ProbeOutcome:
    return ProbeOutcome(
        ProbeResult.OFFLINE, StreamInfo(url=url, live_status="was_live"), confirmed_offline=True, reason="live_status=was_live"
    )


async def test_trusted_confirmed_offline_deletes_at_threshold(tmp_path, caplog):
    server = RestartStubServer(incoming=[YT_URL])
    prober = ScriptedProber([confirmed_offline(YT_URL)], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, offline_threshold=2)
    watcher._schedules[YT_URL] = _UrlSchedule()

    with caplog.at_level("INFO", logger="live_transcript_cloud_worker.watcher"):
        await watcher._check_url(YT_URL)
        assert server.deleted == []
        assert watcher._schedules[YT_URL].offline_count == 1
        await watcher._check_url(YT_URL)

    assert server.deleted == [YT_URL]
    assert YT_URL not in watcher._schedules
    assert any("confirmed offline 2x" in r.getMessage() for r in caplog.records)


async def test_degraded_confirmed_offline_at_threshold_does_not_delete(tmp_path):
    # A failed DELETE left offline_count at the threshold; then the jar died.
    # The next UserNotLive arrives confirmed_offline=True but cannot be
    # trusted, so the URL is parked instead of deleted.
    server = RestartStubServer(incoming=[YT_URL])
    prober = ScriptedProber([confirmed_offline(YT_URL)], _tracker(CookieAuth.ROTATED))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, offline_threshold=2)
    schedule = _UrlSchedule()
    schedule.offline_count = 2
    watcher._schedules[YT_URL] = schedule

    await watcher._check_url(YT_URL)

    assert server.deleted == []
    assert schedule.offline_count == 2
    assert schedule.parked
    assert schedule.inconclusive_count == 1


async def test_trusted_offline_wins_over_a_pending_give_up(tmp_path, caplog):
    # A trusted verdict resets the inconclusive streak before the give-up is
    # evaluated, so the removal is logged as what it is: confirmed offline.
    server = RestartStubServer(incoming=[YT_URL])
    prober = ScriptedProber([confirmed_offline(YT_URL)], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(
        tmp_path, incoming_mode=True, server=server, prober=prober, offline_threshold=2, inconclusive_threshold=10
    )
    schedule = _UrlSchedule()
    schedule.offline_count = 1
    schedule.inconclusive_count = 10
    watcher._schedules[YT_URL] = schedule

    with caplog.at_level("INFO", logger="live_transcript_cloud_worker.watcher"):
        await watcher._check_url(YT_URL)

    assert server.deleted == [YT_URL]
    assert schedule.inconclusive_count == 0
    assert not any("giving up" in r.getMessage() for r in caplog.records)
    assert any("confirmed offline 2x" in r.getMessage() for r in caplog.records)


async def test_healthy_but_unrecognised_failures_also_give_up(tmp_path):
    # Cookies are fine; the video is private (or removed). The URL can never
    # resolve, so it must not sit in the queue until the server's 24 h TTL.
    server = RestartStubServer(incoming=[YT_URL])
    private = ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=YT_URL), reason="ERROR: [youtube] abc: Private video.")
    prober = ScriptedProber([private], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, inconclusive_threshold=4)
    watcher._schedules[YT_URL] = _UrlSchedule()

    for _ in range(4):
        await watcher._check_url(YT_URL)

    assert server.deleted == [YT_URL]


async def test_error_probes_never_give_up_on_a_url(tmp_path):
    # A broken yt-dlp binary or a CPU-starved VPS must never edit the
    # operator's queue.
    server = RestartStubServer(incoming=[YT_URL])
    error = ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=YT_URL), reason="could not run yt-dlp")
    prober = ScriptedProber([error], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, inconclusive_threshold=2)
    watcher._schedules[YT_URL] = _UrlSchedule()

    for _ in range(10):
        await watcher._check_url(YT_URL)

    assert server.deleted == []
    assert watcher._schedules[YT_URL].inconclusive_count == 0


async def test_static_mode_never_gives_up(tmp_path):
    server = RestartStubServer()
    prober = ScriptedProber([bot_check(YT_URL)], _tracker(CookieAuth.ABSENT))
    watcher = make_restart_watcher(tmp_path, incoming_mode=False, server=server, prober=prober, inconclusive_threshold=2)
    watcher._schedules[YT_URL] = _UrlSchedule()

    for _ in range(10):
        await watcher._check_url(YT_URL)

    assert server.deleted == []
    assert YT_URL in watcher._schedules


async def test_live_verdict_resets_the_inconclusive_streak(tmp_path):
    # Via the blacklist branch so no capture runs.
    server = RestartStubServer(incoming=[YT_URL])
    live = ProbeOutcome(ProbeResult.LIVE, StreamInfo(url=YT_URL, stream_id="abc", is_live=True, live_status="is_live"))
    prober = ScriptedProber([live], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, id_blacklist=("abc",))
    schedule = _UrlSchedule()
    schedule.inconclusive_count = 9
    schedule.parked = True
    watcher._schedules[YT_URL] = schedule

    await watcher._check_url(YT_URL)

    assert schedule.inconclusive_count == 0
    assert not schedule.parked
    assert server.deleted == []


async def test_replaced_jar_wakes_parked_urls_only(tmp_path):
    prober = ScriptedProber([], _tracker(CookieAuth.ABSENT))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=RestartStubServer(), prober=prober)
    parked, normal = _UrlSchedule(), _UrlSchedule()
    parked.parked = True
    parked.next_check = normal.next_check = time.time() + 600
    watcher._schedules[YT_URL] = parked
    watcher._schedules["https://www.twitch.tv/x"] = normal

    watcher._wake_parked_if_jar_replaced()
    assert parked.next_check > time.time()  # nothing changed yet

    prober.jar_generation += 1
    watcher._wake_parked_if_jar_replaced()
    assert parked.next_check == 0.0
    assert normal.next_check > time.time()

    parked.next_check = time.time() + 600
    watcher._wake_parked_if_jar_replaced()
    assert parked.next_check > time.time()  # a generation is acted on once


# ------------------------------------------------- post-capture verify path


def _live(url: str) -> ProbeOutcome:
    return ProbeOutcome(ProbeResult.LIVE, StreamInfo(url=url, stream_id="s1", is_live=True, live_status="is_live"))


async def _capture_watcher(tmp_path, url, verify, tracker):
    server = RestartStubServer(incoming=[url])
    prober = ScriptedProber([_live(url), verify], tracker)
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober)
    watcher._schedules[url] = _UrlSchedule()

    async def fake_run_stream(info):
        return True  # capture ran and ended on its own

    watcher._run_stream = fake_run_stream
    return watcher, server, prober


async def test_ended_stream_is_dropped_right_after_capture(tmp_path):
    verify = ProbeOutcome(
        ProbeResult.OFFLINE, StreamInfo(url=YT_URL, live_status="was_live"), confirmed_offline=True, reason="live_status=was_live"
    )
    watcher, server, prober = await _capture_watcher(tmp_path, YT_URL, verify, _tracker(CookieAuth.OK))

    await watcher._check_url(YT_URL)

    assert server.deleted == [YT_URL]
    assert YT_URL not in watcher._schedules
    assert prober.calls == [YT_URL, YT_URL]


def _assert_recheck_scheduled(schedule: _UrlSchedule, before: float) -> None:
    assert before + POST_CAPTURE_RECHECK_SECONDS <= schedule.next_check <= time.time() + POST_CAPTURE_RECHECK_SECONDS + 0.5


async def test_bot_checked_verify_after_capture_does_not_drop(tmp_path):
    # The jar died during the capture: the verify probe is blind. The stream
    # may well still be live, so the URL stays and the counters decide.
    watcher, server, _ = await _capture_watcher(tmp_path, YT_URL, bot_check(YT_URL), _tracker(CookieAuth.ROTATED))

    before = time.time()
    await watcher._check_url(YT_URL)

    assert server.deleted == []
    _assert_recheck_scheduled(watcher._schedules[YT_URL], before)


@pytest.mark.parametrize("state", [CookieAuth.ROTATED, CookieAuth.ABSENT])
async def test_dead_jar_confirmed_offline_verify_after_capture_does_not_drop(tmp_path, state):
    # A signed-out probe of a YouTube URL raises UserNotLive, which reads as
    # confirmed_offline=True regardless of cookie state. Only the degraded
    # guard on the fast path keeps the operator's URL from being deleted.
    verify = ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=YT_URL), confirmed_offline=True, reason=YT_OFFLINE)
    watcher, server, prober = await _capture_watcher(tmp_path, YT_URL, verify, _tracker(state))

    before = time.time()
    await watcher._check_url(YT_URL)

    assert prober.calls == [YT_URL, YT_URL]  # the verify probe ran; the guard kept the URL
    assert server.deleted == []
    schedule = watcher._schedules[YT_URL]
    _assert_recheck_scheduled(schedule, before)
    assert schedule.offline_count == 0 and schedule.inconclusive_count == 0 and not schedule.parked


async def test_error_verify_after_capture_does_not_drop(tmp_path):
    verify = ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=YT_URL), reason="probe timed out after 30s")
    watcher, server, _ = await _capture_watcher(tmp_path, YT_URL, verify, _tracker(CookieAuth.OK))

    before = time.time()
    await watcher._check_url(YT_URL)

    assert server.deleted == []
    _assert_recheck_scheduled(watcher._schedules[YT_URL], before)


async def test_failed_fast_path_delete_is_retried_on_the_next_trusted_probe(tmp_path):
    class FlakyDeleteServer(RestartStubServer):
        async def delete_incoming(self, key, url):
            self.deleted.append(url)
            return len(self.deleted) > 1  # the first DELETE fails

    server = FlakyDeleteServer(incoming=[YT_URL])
    prober = ScriptedProber([_live(YT_URL), confirmed_offline(YT_URL)], _tracker(CookieAuth.OK))
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, offline_threshold=6)
    watcher._schedules[YT_URL] = _UrlSchedule()

    async def fake_run_stream(info):
        return True

    watcher._run_stream = fake_run_stream
    await watcher._check_url(YT_URL)  # capture, verify=was_live, DELETE fails
    assert server.deleted == [YT_URL]
    assert YT_URL in watcher._schedules

    await watcher._check_url(YT_URL)  # the 5 s recheck: another trusted offline
    assert server.deleted == [YT_URL, YT_URL]
    assert YT_URL not in watcher._schedules


async def test_twitch_verify_drops_even_while_youtube_cookies_are_dead(tmp_path):
    url = "https://www.twitch.tv/somechan"
    verify = ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=url), confirmed_offline=True, reason=TWITCH_OFFLINE)
    watcher, server, _ = await _capture_watcher(tmp_path, url, verify, _tracker(CookieAuth.ROTATED))

    await watcher._check_url(url)

    assert server.deleted == [url]


# ------------------------------------ queue re-read before spending a probe


class QueueServer(RestartStubServer):
    """get_incoming answers from ``incoming`` and can stop the watcher."""

    def __init__(self, incoming=(), stop_event=None, answer=True):
        super().__init__(incoming=incoming)
        self.incoming_calls = 0
        self.stop_event = stop_event
        self.answer = answer

    async def get_incoming(self, key):
        self.incoming_calls += 1
        if self.stop_event is not None:
            self.stop_event.set()
        return list(self.incoming) if self.answer else None


async def test_stale_queue_view_is_re_read_before_a_probe(tmp_path):
    # The operator removed the URL on the admin page. The events signal never
    # says so (it only announces additions), so the worker must re-read the
    # queue itself before spending a probe on the URL.
    stop = asyncio.Event()
    server = QueueServer(incoming=[], stop_event=stop)
    prober = ScriptedProber([bot_check(YT_URL)], _tracker(CookieAuth.OK), stop_event=stop, stop_after=1)
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, incoming_interval=15, stop_event=stop)
    watcher._schedules[YT_URL] = _UrlSchedule()  # due now
    watcher._last_incoming_refresh = time.time() - 20  # older than interval_seconds, younger than the fallback

    await watcher.run()

    assert server.incoming_calls == 1
    assert prober.calls == []
    assert watcher._schedules == {}


async def test_fresh_queue_view_is_not_re_read_before_a_probe(tmp_path):
    stop = asyncio.Event()
    server = QueueServer(incoming=[YT_URL], stop_event=stop)
    prober = ScriptedProber([bot_check(YT_URL)], _tracker(CookieAuth.OK), stop_event=stop, stop_after=1)
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, incoming_interval=15, stop_event=stop)
    watcher._schedules[YT_URL] = _UrlSchedule()
    watcher._last_incoming_refresh = time.time()

    await watcher.run()

    assert server.incoming_calls == 0
    assert prober.calls == [YT_URL]


async def test_queue_re_read_is_skipped_while_a_retry_backoff_is_pending(tmp_path):
    # _defer_incoming_retry rewinds the refresh timestamp; without the
    # failures gate every due probe would issue another failing GET.
    stop = asyncio.Event()
    server = QueueServer(incoming=[YT_URL], stop_event=stop, answer=False)
    prober = ScriptedProber([bot_check(YT_URL)], _tracker(CookieAuth.OK), stop_event=stop, stop_after=1)
    watcher = make_restart_watcher(tmp_path, incoming_mode=True, server=server, prober=prober, incoming_interval=15, stop_event=stop)
    watcher._schedules[YT_URL] = _UrlSchedule()
    watcher._incoming_failures = 1
    watcher._last_incoming_refresh = time.time() - watcher._incoming_interval() + 60  # retry due in 60 s

    await watcher.run()

    assert server.incoming_calls == 0
    assert prober.calls == [YT_URL]
    assert watcher._incoming_failures == 1
