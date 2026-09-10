"""yt-dlp invocation helpers: auth args and the liveness probe.

The probe (``yt-dlp -j``) is expensive, high CPU, seconds of wall time, so
callers rate-limit it through a process-wide semaphore and the watcher's
adaptive per-URL schedule (docs/07 §7).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from .config import Config
from .cookieauth import CookieAuthTracker, classify_cookie_auth
from .models import Platform, ProbeResult, StreamInfo, valid_stream_id
from .state import ChannelState

logger = logging.getLogger(__name__)

# --verbose is what makes cookie health observable (see cookieauth), but it
# prepends ~16 lines of banner to every probe's stderr. Those lines would
# otherwise crowd out the real error in the 4 KB stream stats log and in the
# probe failure log line, so they are stripped everywhere except the cookie
# classifier, which is the one consumer that wants them.
_DEBUG_LINE_RE = re.compile(r"^\[debug\] .*\n?", re.MULTILINE)


def _terse(stderr: str) -> str:
    """yt-dlp stderr with the --verbose banner removed."""
    return _DEBUG_LINE_RE.sub("", stderr).strip()


_BEGIN_IN_RE = re.compile(r"will begin in ([^.]+)")
_DURATION_PART_RE = re.compile(r"(\d+)\s+(day|hour|minute|second)s?")
_UNIT_SECONDS = {"day": 86400, "hour": 3600, "minute": 60, "second": 1}

# yt-dlp stamps live titles with the time the metadata was fetched
# ("<title> 2026-08-12 17:38"), which would otherwise churn the stream title
# on every reactivation. Only *trailing* stamps are stripped: a date in the
# middle (or at the start) of a title is the broadcaster's own text.
_STAMP = r"""
    (?:
        \d{4}-\d{2}-\d{2}                     # 2026-08-12
        (?:[\sT,]+\d{1,2}:\d{2}(?::\d{2})?)?  #  optionally 17:38(:09)
      | \d{1,2}/\d{1,2}/\d{2,4}               # 12/08/2026
        (?:[\s,]+\d{1,2}:\d{2}(?::\d{2})?)?
      | \d{1,2}:\d{2}(?::\d{2})?              # bare 17:38
    )
    (?:\s*(?:[AaPp]\.?[Mm]\.?\b|[A-Z]{2,5}\b))?  # optional am/pm or timezone
"""
_SEP = r"[\s\-–—|,]*"
# Bracketed form first ("Title (2026-08-12 17:38)"): the brackets only count
# as part of the stamp when they wrap it, so "... (Part 1) 12:00" keeps its
# parenthesis.
_BRACKETED_STAMP_RE = re.compile(rf"{_SEP}[(\[]\s*{_STAMP}\s*[)\]]\s*$", re.VERBOSE)
# One stamp only: yt-dlp appends exactly one, and a greedy repeat would also
# swallow a time the broadcaster wrote ("karaoke 21:00 JST 2026-08-12 17:38").
_TRAILING_STAMP_RE = re.compile(rf"{_SEP}{_STAMP}{_SEP}$", re.VERBOSE)


def strip_trailing_timestamp(title: str) -> str:
    """Drop the date/time yt-dlp appends to live stream titles.

    A title that is *only* a stamp is left alone: an empty title is worse
    than a noisy one.
    """
    cleaned = title.strip()
    stripped = _BRACKETED_STAMP_RE.sub("", cleaned)
    if stripped == cleaned:
        stripped = _TRAILING_STAMP_RE.sub("", cleaned)
    stripped = stripped.strip()
    return stripped or cleaned


def platform_of(url: str) -> Platform:
    lowered = url.lower()
    if "twitch.tv" in lowered:
        return Platform.TWITCH
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return Platform.YOUTUBE
    return Platform.OTHER


def auth_args(config: Config, url: str, purpose: str) -> list[str]:
    """Extra yt-dlp args for a URL. Twitch takes none: it has no
    ``availability`` field and doesn't need the cookies."""
    if platform_of(url) is Platform.TWITCH:
        return []
    args = ["--match-filter", "availability!=?subscriber_only"]
    cookies = config.cookies_file(purpose)
    if cookies is not None:
        if cookies.is_file():
            args += ["--cookies", str(cookies)]
        else:
            logger.warning("cookies enabled but file missing: %s", cookies)
    return args


class ProbeOutcome:
    """Everything the watcher needs from one liveness check."""

    def __init__(
        self,
        result: ProbeResult,
        info: StreamInfo,
        confirmed_offline: bool = False,
        scheduled_start: int | None = None,
    ) -> None:
        self.result = result
        self.info = info
        self.confirmed_offline = confirmed_offline
        self.scheduled_start = scheduled_start


def _parse_scheduled(stderr: str) -> int | None:
    """Parse "This live event will begin in <duration>." into a unix time."""
    match = _BEGIN_IN_RE.search(stderr)
    if not match:
        return None
    total = 0
    for amount, unit in _DURATION_PART_RE.findall(match.group(1)):
        total += int(amount) * _UNIT_SECONDS[unit]
    return int(time.time()) + total if total > 0 else None


def classify_probe_failure(stderr: str, platform: Platform) -> tuple[int | None, bool]:
    """(scheduled_start, confirmed_offline) from a failed probe's stderr.

    Scheduled-start parsing is YouTube-shaped ("This live event will begin
    in ..."), so it is skipped for Twitch. The offline confirmation text is
    yt-dlp's shared UserNotLive error ("... is not currently live") and is
    emitted for every platform, Twitch included.
    """
    if platform is not Platform.TWITCH:
        scheduled = _parse_scheduled(stderr)
        if scheduled:
            return scheduled, False
    return None, "not currently live" in stderr


def _info_from_metadata(url: str, metadata: dict) -> StreamInfo:
    platform = platform_of(url)
    is_live = bool(metadata.get("is_live", False)) or metadata.get("live_status", "") == "is_live"
    live_status = str(metadata.get("live_status", "") or ("is_live" if is_live else "unknown"))

    if platform is Platform.TWITCH:
        start = metadata.get("timestamp")
        title = f"{metadata.get('display_id', '')} - {metadata.get('description', '')}".strip(" -")
    else:
        start = metadata.get("release_timestamp") or metadata.get("timestamp")
        title = str(metadata.get("title", ""))

    scheduled = None
    if live_status == "is_upcoming":
        scheduled = metadata.get("release_timestamp")

    return StreamInfo(
        url=url,
        stream_id=str(metadata.get("id", "")),
        title=strip_trailing_timestamp(title) or "untitled",
        is_live=is_live,
        live_status=live_status,
        scheduled_start=int(scheduled) if scheduled else None,
        start_time=int(start) if start else None,
    )


class Prober:
    """Bounded-concurrency yt-dlp -j liveness probing."""

    def __init__(self, config: Config, cookies: CookieAuthTracker | None = None) -> None:
        self.config = config
        self.cookies = cookies or CookieAuthTracker()
        self._semaphore = asyncio.Semaphore(config.capture.probe_concurrency)

    async def probe(self, url: str, state: ChannelState | None = None) -> ProbeOutcome:
        cmd = [
            self.config.capture.yt_dlp_path,
            "-j",
            # --verbose buys the only cookie-health signal that is a level
            # rather than a one-shot edge: the YouTube extractor's "Found
            # YouTube account cookies", printed during init before the first
            # request. It goes to stderr only, so the -j JSON on stdout is
            # untouched.
            "-v",
            *auth_args(self.config, url, "check"),
            url,
        ]
        async with self._semaphore:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=self.config.capture.probe_timeout_seconds)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
                    logger.warning("probe timed out for %s", url)
                    return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url))
            except OSError as exc:
                logger.error("could not run yt-dlp (%s): %s", cmd[0], exc)
                return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url))

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        # Classify on every probe, successful or not. yt-dlp prints the cookie
        # verdict during init and then carries on anonymously, so a probe that
        # SUCCEEDS is just as much a report on the jar as one that fails --
        # and discarding those was why a dead cookie could go unnoticed
        # indefinitely.
        jar = self.config.cookies_file("check")
        self.cookies.observe(
            classify_cookie_auth(
                # Mirror auth_args exactly: a configured-but-missing file means
                # --cookies was never passed, and yt-dlp cannot report on a jar
                # it was not given.
                cookies_enabled=jar is not None and jar.is_file(),
                is_youtube=platform_of(url) is Platform.YOUTUBE,
                stderr=stderr,
            )
        )

        terse = _terse(stderr)
        if state is not None:
            state.stream_stats_log.append(
                f"--- {time.strftime('%Y-%m-%dT%H:%M:%S')} probe {url} rc={proc.returncode}\n{stdout[:4096]}\n{terse[:4096]}"
            )

        if proc.returncode != 0:
            scheduled, confirmed = classify_probe_failure(terse, platform_of(url))
            if scheduled:
                return ProbeOutcome(ProbeResult.UPCOMING, StreamInfo(url=url), scheduled_start=scheduled)
            if not confirmed:
                logger.debug("probe %s exited %s: %s", url, proc.returncode, terse[-500:])
            return ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=url), confirmed_offline=confirmed)

        try:
            metadata = json.loads(stdout)
        except json.JSONDecodeError:
            logger.warning("probe %s produced unparseable JSON", url)
            return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url))

        info = _info_from_metadata(url, metadata)
        if info.is_live:
            if not valid_stream_id(info.stream_id):
                logger.error(
                    "stream %r at %s has an unpublishable ID (must match [A-Za-z0-9_-]{1,128}); skipping",
                    info.stream_id,
                    url,
                )
                return ProbeOutcome(ProbeResult.ERROR, info)
            return ProbeOutcome(ProbeResult.LIVE, info)
        if info.is_upcoming:
            return ProbeOutcome(ProbeResult.UPCOMING, info, scheduled_start=info.scheduled_start)
        return ProbeOutcome(ProbeResult.OFFLINE, info)
