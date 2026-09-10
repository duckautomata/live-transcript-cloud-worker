"""Cookie-auth detection, including a guard against yt-dlp rewording us into
a silent false negative."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from live_transcript_cloud_worker.cookieauth import (
    CONSECUTIVE_TO_ALARM,
    CookieAuth,
    CookieAuthTracker,
    classify_cookie_auth,
)

# Real yt-dlp output, captured from 2026.07.04 with a jar holding LOGIN_INFO
# plus SAPISID.
AUTHENTICATED_STDERR = """[debug] Command-line config: ['-v', '-j', '--cookies', 'cookies.txt', 'https://www.youtube.com/@x/live']
[debug] Loaded 1744 extractors
[debug] [youtube] Found YouTube account cookies
[youtube] Extracting URL: https://www.youtube.com/@x/live
"""

ANONYMOUS_STDERR = """[debug] Command-line config: ['-v', '-j', '--cookies', 'cookies.txt', 'https://www.youtube.com/@x/live']
[debug] Loaded 1744 extractors
[youtube] Extracting URL: https://www.youtube.com/@x/live
ERROR: [youtube] x: The channel is not currently live
"""

ROTATED_STDERR = """[debug] [youtube] Found YouTube account cookies
WARNING: [youtube] The provided YouTube account cookies are no longer valid. They have likely been rotated in the browser as a security measure.
"""


# ------------------------------------------------------------- classification


def test_authenticated_probe_reads_ok():
    assert classify_cookie_auth(True, True, AUTHENTICATED_STDERR) is CookieAuth.OK


def test_anonymous_probe_reads_absent():
    assert classify_cookie_auth(True, True, ANONYMOUS_STDERR) is CookieAuth.ABSENT


def test_rotation_warning_reads_rotated():
    assert classify_cookie_auth(True, True, ROTATED_STDERR) is CookieAuth.ROTATED


def test_twitch_never_contributes():
    # Twitch is never given cookies, so it must not be able to raise or clear
    # the alarm for YouTube's jar.
    assert classify_cookie_auth(True, False, ANONYMOUS_STDERR) is CookieAuth.NA


def test_cookies_disabled_is_not_a_failure():
    assert classify_cookie_auth(False, True, ANONYMOUS_STDERR) is CookieAuth.NA


# ------------------------------------------------------------------- tracking


def test_absent_needs_a_streak_before_alarming():
    tracker = CookieAuthTracker()
    for _ in range(CONSECUTIVE_TO_ALARM - 1):
        tracker.observe(CookieAuth.ABSENT)
        assert not tracker.degraded
    tracker.observe(CookieAuth.ABSENT)
    assert tracker.degraded
    assert tracker.state is CookieAuth.ABSENT


def test_one_good_probe_clears_the_streak():
    # A single torn read of the jar, or one probe that died before extractor
    # init, must not be able to accumulate toward an alarm.
    tracker = CookieAuthTracker()
    for _ in range(CONSECUTIVE_TO_ALARM * 3):
        tracker.observe(CookieAuth.ABSENT)
        tracker.observe(CookieAuth.OK)
    assert not tracker.degraded


def test_rotation_alarms_immediately():
    # yt-dlp only says this when it watched the auth cookies vanish from a jar
    # it had already accepted, so there is nothing to debounce.
    tracker = CookieAuthTracker()
    tracker.observe(CookieAuth.ROTATED)
    assert tracker.degraded


def test_recovery_transitions_back_to_ok():
    tracker = CookieAuthTracker()
    tracker.observe(CookieAuth.ROTATED)
    tracker.observe(CookieAuth.OK)
    assert not tracker.degraded
    assert tracker.state is CookieAuth.OK


def test_na_observations_are_ignored():
    tracker = CookieAuthTracker()
    tracker.observe(CookieAuth.ABSENT)
    for _ in range(10):
        tracker.observe(CookieAuth.NA)
    assert not tracker.degraded  # NA neither advanced nor reset the streak
    tracker.observe(CookieAuth.ABSENT)
    tracker.observe(CookieAuth.ABSENT)
    assert tracker.degraded


# ------------------------------------------------------- upstream drift guard


@pytest.mark.skipif(shutil.which("yt-dlp") is None, reason="yt-dlp not installed")
def test_ytdlp_still_emits_the_string_we_match_on(tmp_path):
    """The detector is a substring match on yt-dlp's verbose output, and the
    Dockerfile resolves the yt-dlp version at build time. If upstream rewords
    that line, every probe silently reads as authenticated and the whole
    feature goes quiet -- exactly the failure it exists to prevent. Fail here
    instead.

    Runs against a dead proxy, so it makes no request to Google: the line is
    printed during extractor init, before the first fetch.
    """
    jar = tmp_path / "jar.txt"
    jar.write_text(
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t1900000000\tLOGIN_INFO\tnotarealvalue\n"
        ".youtube.com\tTRUE\t/\tTRUE\t1900000000\tSAPISID\tnotarealvalue\n"
    )
    result = subprocess.run(
        [
            "yt-dlp",
            "-v",
            "-j",
            "--proxy",
            "http://127.0.0.1:1",
            "--cookies",
            str(jar),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert classify_cookie_auth(True, True, result.stderr) is CookieAuth.OK, (
        "yt-dlp no longer prints the authenticated-cookies line this detector "
        f"matches on; update _AUTH_OK_RE in cookieauth.py. stderr was:\n{result.stderr[:2000]}"
    )


@pytest.mark.skipif(shutil.which("yt-dlp") is None, reason="yt-dlp not installed")
def test_ytdlp_stays_silent_for_a_jar_with_no_auth_cookies(tmp_path):
    """The other half of the drift guard: a jar with no account cookies must
    NOT produce the line, or the detector would read every probe as healthy."""
    jar = tmp_path / "jar.txt"
    jar.write_text("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t1900000000\tPREF\tnotarealvalue\n")
    result = subprocess.run(
        [
            "yt-dlp",
            "-v",
            "-j",
            "--proxy",
            "http://127.0.0.1:1",
            "--cookies",
            str(jar),
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert classify_cookie_auth(True, True, result.stderr) is CookieAuth.ABSENT
