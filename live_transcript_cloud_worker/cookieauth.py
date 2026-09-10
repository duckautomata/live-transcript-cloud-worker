"""Worker-global YouTube cookie-auth health, read out of yt-dlp's own verbose
output rather than by parsing the Netscape jar ourselves.

Why yt-dlp's verdict and not our own parse of the jar:

* It is the predicate yt-dlp actually authenticates with -- LOGIN_INFO present
  plus one of SAPISID / __Secure-1PAPISID / __Secure-3PAPISID, over the cookies
  it resolves for https://www.youtube.com. A hand-rolled parser has to track
  that predicate forever, and has to get the domain scoping right: the same
  cookie names can appear on .google.com, which yt-dlp never consults.
* It is a LEVEL, re-derived from the jar on every invocation, so it survives a
  worker restart. The rotation WARNING does not: yt-dlp only emits that when it
  watched the auth cookies vanish from a jar it had already accepted, so a jar
  that was already dead when the process started never produces it. That is the
  steady state after the first rotation, because the run that observes it writes
  the cleared jar back over cookies.txt on exit.
* It cannot be fooled by a torn read. yt-dlp rewrites the jar with a plain
  truncating write, so a separate reader races it; asking yt-dlp does not,
  because a yt-dlp that loaded a torn jar really did run unauthenticated.
* It costs nothing. The line is printed during extractor init, before the first
  request, so no extra traffic ever reaches the burner account.

Deliberately NOT based on cookie expiry. yt-dlp never prunes expired cookies,
and the failure mode is YouTube deleting LOGIN_INFO rather than anything
lapsing, so every surviving entry still reads far-future while the jar is
already dead. Expiry is the wrong field, not merely a late one.
"""

from __future__ import annotations

import logging
import re
import time
from enum import StrEnum

logger = logging.getLogger(__name__)

# Match the stable substrings only, never the surrounding formatting: yt-dlp
# prefixes these with "[debug] [youtube] " / "WARNING: [youtube] ". Both are
# pinned by tests/test_cookieauth.py, because the Dockerfile resolves the
# yt-dlp version at build time and an upstream reword would otherwise turn
# every probe into a silent false negative -- the exact failure this module
# exists to prevent.
_AUTH_OK_RE = re.compile(r"Found YouTube account cookies")
_ROTATED_RE = re.compile(r"YouTube account cookies are no longer valid")

# Consecutive cookie-less YouTube probes before the state flips. Three, so that
# a single torn read of the jar, a probe that died before extractor init, or a
# restart window cannot raise an alarm on its own.
CONSECUTIVE_TO_ALARM = 3


class CookieAuth(StrEnum):
    NA = "na"  # cookies not in play here (Twitch, or cookies disabled)
    OK = "ok"  # yt-dlp loaded a jar with usable account cookies
    ABSENT = "absent"  # --cookies was passed, yt-dlp found no usable auth in it
    ROTATED = "rotated"  # YouTube invalidated the jar mid-request


def classify_cookie_auth(cookies_enabled: bool, is_youtube: bool, stderr: str) -> CookieAuth:
    """What yt-dlp's verbose stderr says about the jar it just loaded.

    Only meaningful when --cookies was actually passed. Twitch never gets
    cookies, and the debug line comes from YouTube's extractor alone, so
    neither may contribute to the state.
    """
    if not is_youtube or not cookies_enabled:
        return CookieAuth.NA
    if _ROTATED_RE.search(stderr):
        return CookieAuth.ROTATED
    if _AUTH_OK_RE.search(stderr):
        return CookieAuth.OK
    return CookieAuth.ABSENT


class CookieAuthTracker:
    """Debounced, worker-global cookie health.

    Worker-global rather than per-channel because one jar backs every YouTube
    channel. One instance per process; the worker is single-threaded asyncio,
    so no locking is needed.
    """

    def __init__(self, threshold: int = CONSECUTIVE_TO_ALARM) -> None:
        self.threshold = threshold
        self.state = CookieAuth.NA
        self.since = int(time.time())
        self.reason = ""
        self._bad_streak = 0

    @property
    def degraded(self) -> bool:
        """True when yt-dlp is scraping YouTube anonymously.

        Read by the watcher so a cookie outage cannot drain the incoming queue.
        """
        return self.state in (CookieAuth.ABSENT, CookieAuth.ROTATED)

    def observe(self, auth: CookieAuth) -> None:
        if auth is CookieAuth.NA:
            return
        if auth is CookieAuth.OK:
            self._bad_streak = 0
            self._transition(CookieAuth.OK, "")
            return
        self._bad_streak += 1
        if auth is CookieAuth.ROTATED:
            # yt-dlp only says this when it watched the auth cookies vanish
            # from a jar it had already accepted. Nothing benign produces it,
            # so it does not wait out the streak.
            self._transition(CookieAuth.ROTATED, "YouTube invalidated the cookies mid-request")
        elif self._bad_streak >= self.threshold:
            self._transition(
                CookieAuth.ABSENT,
                f"no usable account cookies in the jar ({self._bad_streak} consecutive probes)",
            )

    def _transition(self, state: CookieAuth, reason: str) -> None:
        if state is self.state:
            return
        previous, self.state = self.state, state
        self.since, self.reason = int(time.time()), reason
        if state is CookieAuth.OK:
            if previous in (CookieAuth.ABSENT, CookieAuth.ROTATED):
                logger.warning("youtube cookies are authenticating again (was %s)", previous.value)
            return
        # ERROR rather than WARNING: the JSON file handler stamps the level and
        # promtail labels it, so this is directly alertable in Loki without any
        # server-side change.
        logger.error(
            "youtube cookies are not authenticating: %s. yt-dlp is scraping anonymously, so "
            "streams may read as offline. Mint a new cookies.txt from the burner account.",
            reason,
            extra={"cookie_state": state.value, "cookie_streak": self._bad_streak},
        )
