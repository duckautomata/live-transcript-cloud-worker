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
from dataclasses import dataclass

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


_BEGIN_IN_RE = re.compile(r"(?:will begin in|premieres? in) ([^.]+)", re.IGNORECASE)
_DURATION_PART_RE = re.compile(r"(\d+)\s+(day|hour|minute|second)s?")
_UNIT_SECONDS = {"day": 86400, "hour": 3600, "minute": 60, "second": 1}
# YouTube's playability reasons for a broadcast that has not started. "This
# live event will begin in 3 hours." carries a duration; "... will begin in a
# few moments." (the streamer is late) and premieres ("Premieres in 2 hours")
# do not always parse to one. Every one of them is a positive "upcoming"
# verdict and must never read as a failure the watcher could give up on.
_UPCOMING_RE = re.compile(r"will begin (?:in|shortly)|premieres? in", re.IGNORECASE)

# yt-dlp failures that describe this worker's network, its IP or its own
# extractor rather than the stream: YouTube unreachable, a rate limit, a 5xx,
# an extractor broken by a YouTube page change. They hit every URL on every
# channel at once, so they are ERROR outcomes that never move the operator's
# queue -- unlike a bot check, a private video or an unknown status, which
# are verdicts on the URL. Matched against the ERROR line only: the cause
# text, never yt-dlp's generic "Unable to download webpage" prefix, which it
# also prints for a per-URL HTTP 404 (a handle that does not exist).
_WORKER_SIDE_RE = re.compile(
    r"TransportError|urlopen error|Connection refused|Connection reset|Name or service not known|Network is unreachable"
    r"|Temporary failure in name resolution|timed out|HTTP Error (?:429|5\d\d)|Too Many Requests|has been rate-limited"
    r"|This content isn't available, try again later|Unable to extract|Failed to extract any player response"
    r"|All player responses are invalid|IP is likely being blocked",
    re.IGNORECASE,
)

# The cookies whose values only change when the operator mints a new jar (or
# YouTube invalidates the session): yt-dlp's own exit-time rewrite of a jar
# keeps them byte-identical while churning the per-request ones (YSC,
# VISITOR_INFO1_LIVE, the *PSIDTS timestamps). Their values are the jar's
# identity for replacement detection; see Prober.poll_jar.
_AUTH_COOKIE_NAMES = frozenset(
    {"LOGIN_INFO", "SAPISID", "__Secure-1PAPISID", "__Secure-3PAPISID", "SID", "__Secure-1PSID", "__Secure-3PSID", "HSID", "SSID", "APISID"}
)
_FILTER_SKIP_RE = re.compile(r"^\[download\] .*does not pass filter.*$", re.MULTILINE)

# A failed probe's reason is one line, long enough to keep yt-dlp's message
# and short enough for a log line.
_REASON_MAX = 240

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
        reason: str = "",
    ) -> None:
        self.result = result
        self.info = info
        self.confirmed_offline = confirmed_offline
        self.scheduled_start = scheduled_start
        # One human-readable line for a probe that did not resolve to a
        # stream: yt-dlp's ERROR line, "probe timed out", "live_status=unknown".
        # It is what the watcher prints when it parks or gives up on a URL.
        self.reason = reason

    @property
    def inconclusive(self) -> bool:
        """A verdict that says nothing about the stream: yt-dlp failed for a
        reason other than "not live" / "upcoming" (a private or removed video,
        a bot check), resolved the page without a usable live_status, or
        printed no metadata at all (an entry the --match-filter rejected,
        i.e. members-only content). ERROR is deliberately excluded: a
        timeout, a missing binary, garbage output, or a yt-dlp failure on
        this worker's side (network, rate limit, broken extractor) describes
        the worker, not the URL, and must never move the operator's queue.
        """
        return self.result is ProbeResult.OFFLINE and not self.confirmed_offline


def _parse_scheduled(stderr: str) -> int | None:
    """Parse "This live event will begin in <duration>." (or "Premieres in
    <duration>") into a unix time; None when no duration is given."""
    match = _BEGIN_IN_RE.search(stderr)
    if not match:
        return None
    total = 0
    for amount, unit in _DURATION_PART_RE.findall(match.group(1)):
        total += int(amount) * _UNIT_SECONDS[unit]
    return int(time.time()) + total if total > 0 else None


@dataclass(frozen=True)
class FailureVerdict:
    """What a failed (rc != 0) probe's stderr says about the stream."""

    upcoming: bool = False
    scheduled_start: int | None = None  # unix seconds, when the message carried a duration
    confirmed_offline: bool = False
    worker_side: bool = False  # the failure is about this worker's network/IP/extractor, not the URL


def classify_probe_failure(stderr: str, platform: Platform) -> FailureVerdict:
    """Classify a failed probe from its stderr.

    Upcoming detection is YouTube-shaped ("This live event will begin in
    ...", "Premieres in ..."), so it is skipped for Twitch. A recognised
    upcoming message without a parseable duration ("will begin in a few
    moments") is still upcoming, not a failure: the streamer is merely late.
    The offline confirmation text is yt-dlp's shared UserNotLive error
    ("... is not currently live") and is emitted for every platform, Twitch
    included. A failure on this worker's side (network, rate limit, broken
    extractor) is flagged ``worker_side`` so the watcher grades it ERROR.
    Anything else is an unrecognised failure, which the watcher treats as
    inconclusive.
    """
    if platform is not Platform.TWITCH and _UPCOMING_RE.search(stderr):
        return FailureVerdict(upcoming=True, scheduled_start=_parse_scheduled(stderr))
    if "not currently live" in stderr:
        return FailureVerdict(confirmed_offline=True)
    # The ERROR line only: retry WARNINGs above it ("Unable to download
    # webpage ... Giving up after 3 retries") must not relabel a per-URL
    # verdict, and an empty stderr is not a verdict at all.
    return FailureVerdict(worker_side=bool(_WORKER_SIDE_RE.search(_error_reason(stderr))))


def _error_reason(terse: str) -> str:
    """The one line of a failed probe worth showing an operator.

    With --verbose yt-dlp follows its ERROR line with a Python traceback, so
    the *last* line of stderr is ``raise ExtractorError(...)``; the ERROR
    line itself is the verdict.
    """
    lines = [line.strip() for line in terse.splitlines() if line.strip()]
    for prefix in ("ERROR:", "WARNING:"):
        for line in reversed(lines):
            if line.startswith(prefix):
                return line[:_REASON_MAX]
    return lines[-1][:_REASON_MAX] if lines else "yt-dlp printed nothing"


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
        # Jar replacement detection, see poll_jar().
        self.jar_generation = 0
        self._jar_signature = self._jar_identity()
        self._jar_busy = 0  # probe subprocesses in flight

    def _jar_identity(self) -> tuple[tuple[str, str, str], ...] | None:
        """The check jar's auth cookies as (domain, name, value), None when
        cookies are disabled or the file is unreadable.

        Content, not mtime/size: yt-dlp rewrites the whole file on every
        normal exit (probe and download alike) with fresh per-request
        cookies, so any timestamp-based signature races our own processes.
        The auth cookies survive such a rewrite unchanged, and a jar the
        operator minted afresh carries different values.
        """
        jar = self.config.cookies_file("check")
        if jar is None:
            return None
        try:
            text = jar.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        found = []
        for line in text.splitlines():
            # Netscape format: HttpOnly cookies (LOGIN_INFO, SID, ...) are
            # written as "#HttpOnly_<domain>\t..." and are real entries.
            if line.startswith("#") and not line.startswith("#HttpOnly_"):
                continue
            fields = line.split("\t")
            if len(fields) >= 7 and fields[5] in _AUTH_COOKIE_NAMES:
                found.append((fields[0].removeprefix("#HttpOnly_"), fields[5], fields[6]))
        return tuple(sorted(found))

    def poll_jar(self) -> int:
        """Bump and return ``jar_generation`` when the operator changed the
        auth cookies in cookies.txt (or YouTube invalidated them).

        Our own yt-dlp runs rewrite the file but leave the auth cookies as
        they were, so they never count. While a probe is in flight this
        returns the current generation without comparing, so a read that
        lands inside the rewrite's truncate-then-write window cannot see a
        torn jar; the identity is re-read once the probe returns. A
        download's rewrite is not guarded (its window is microseconds a few
        times a day), and a false wake only costs one probe round: the
        watcher never counts a wake-triggered probe toward the give-up. The
        watchers poll this each tick and re-probe parked URLs the moment an
        operator overwrites cookies.txt, instead of waiting out the degraded
        cadence.
        """
        if self._jar_busy:
            return self.jar_generation
        current = self._jar_identity()
        if current != self._jar_signature:
            self._jar_signature = current
            self.jar_generation += 1
        return self.jar_generation

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
        # The "[key] " prefix lifts the channel into the JSON log's key
        # field, so a stuck URL can be filtered per channel.
        prefix = f"[{state.key}] " if state is not None else ""
        async with self._semaphore:
            self._jar_busy += 1
            try:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except OSError as exc:
                    logger.error("could not run yt-dlp (%s): %s", cmd[0], exc)
                    return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url), reason=f"could not run yt-dlp: {exc}")
                timeout = self.config.capture.probe_timeout_seconds
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
                    logger.warning("%sprobe timed out for %s", prefix, url)
                    return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url), reason=f"probe timed out after {timeout:g}s")
            finally:
                # Re-read the identity before releasing the in-flight guard
                # so no tick can compare against a torn jar (see poll_jar).
                self._jar_signature = self._jar_identity()
                self._jar_busy -= 1

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        rc = proc.returncode if proc.returncode is not None else 1  # set once communicate() returns

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
            state.stream_stats_log.append(f"--- {time.strftime('%Y-%m-%dT%H:%M:%S')} probe {url} rc={rc}\n{stdout[:4096]}\n{terse[:4096]}")

        if rc < 0 or (rc != 0 and not terse):
            # Killed by a signal (the OOM killer, a stop), or died without a
            # word: nothing here is a verdict on the URL.
            reason = f"yt-dlp killed by signal {-rc}" if rc < 0 else "yt-dlp printed nothing"
            logger.warning("%sprobe %s failed on this worker's side: %s", prefix, url, reason)
            return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url), reason=reason)

        if rc != 0:
            verdict = classify_probe_failure(terse, platform_of(url))
            reason = _error_reason(terse)
            if verdict.upcoming:
                return ProbeOutcome(ProbeResult.UPCOMING, StreamInfo(url=url), scheduled_start=verdict.scheduled_start, reason=reason)
            if verdict.worker_side:
                # Loud on purpose: an outage on our side is worth a WARN line
                # per probe, and it must never look like a verdict on the URL.
                logger.warning("%sprobe %s failed on this worker's side: %s", prefix, url, reason)
                return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url), reason=reason)
            if not verdict.confirmed_offline:
                logger.debug("%sprobe %s exited %s: %s", prefix, url, rc, reason)
            return ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=url), confirmed_offline=verdict.confirmed_offline, reason=reason)

        if not stdout.strip():
            if not terse:
                # Exit 0 without a single non-debug line is not yt-dlp
                # talking about the URL; it is a wrong binary or a wrapper
                # that swallowed its output.
                logger.warning("%sprobe %s produced no output at all", prefix, url)
                return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url), reason="yt-dlp printed nothing")
            # yt-dlp resolved the URL but printed no metadata: the entry was
            # rejected by --match-filter (members-only content the burner
            # account can see but must not pull) or resolved to nothing. A
            # verdict on the URL, not on this worker, so it counts as
            # inconclusive rather than ERROR and the give-up bounds it. The
            # filter's own line names the cause; it is a plain line, so it
            # would otherwise lose to any WARNING printed above it.
            skipped = _FILTER_SKIP_RE.search(terse)
            reason = "no metadata: " + (skipped.group(0).strip()[:_REASON_MAX] if skipped else _error_reason(terse))
            logger.debug("%sprobe %s produced no metadata: %s", prefix, url, reason)
            return ProbeOutcome(ProbeResult.OFFLINE, StreamInfo(url=url), reason=reason)
        try:
            metadata = json.loads(stdout)
        except json.JSONDecodeError:
            logger.warning("%sprobe %s produced unparseable JSON", prefix, url)
            return ProbeOutcome(ProbeResult.ERROR, StreamInfo(url=url), reason="unparseable yt-dlp output")

        info = _info_from_metadata(url, metadata)
        if info.is_live:
            if not valid_stream_id(info.stream_id):
                logger.error(
                    "stream %r at %s has an unpublishable ID (must match [A-Za-z0-9_-]{1,128}); skipping",
                    info.stream_id,
                    url,
                )
                return ProbeOutcome(ProbeResult.ERROR, info, reason="unpublishable stream id")
            return ProbeOutcome(ProbeResult.LIVE, info)
        if info.is_upcoming:
            return ProbeOutcome(ProbeResult.UPCOMING, info, scheduled_start=info.scheduled_start)
        # Resolved successfully but not live. A terminal live_status (an ended
        # broadcast or a plain, never-live video) is as definitive as yt-dlp's
        # UserNotLive failure, so it is reported as confirmed offline: without
        # this an ended YouTube stream -- which reads as "was_live", never
        # "not currently live" -- would be re-probed at the base cadence
        # forever and, in incoming mode, never reach the offline-delete
        # threshold. An empty/unknown status stays soft (unconfirmed).
        return ProbeOutcome(ProbeResult.OFFLINE, info, confirmed_offline=info.is_terminal_offline, reason=f"live_status={info.live_status}")
