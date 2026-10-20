"""
Segment sources for the YouTube Timeline capture (capture_timeline).

A SegmentSource hands the capture one SegmentGroup per sequence number: every
track's segment for that sequence, already stamped with its own clock by
youtube_timeline.inspect_segment. The capture never cares how the bytes were
fetched, so a new transport is a new SegmentSource subclass plus a branch in
create_segment_source():

- HlsSegmentSource (today): polls the audio HLS media playlist that yt-dlp
  resolves and downloads both tracks' segments itself, by sequence number.
- SABR (once yt-dlp ships it): yt-dlp is a standalone binary here, not a
  library, so a SabrSegmentSource would most likely run yt-dlp
  (config.capture.yt_dlp_path) per format (SourceContext carries the watch URL,
  the media type, the per-channel log and the stop check), split its
  fragmented-MP4 output at moof boundaries into TrackSegments (inspect_segment
  reads their clock from tfdt; YouTube's live fragments carry their own moov),
  and pair audio and video by sequence number. Such a source refreshes its own
  URLs, so it reports expires_at None and the capture leaves refreshing to it.
  resolve_stream() keeps each format's raw yt-dlp dict (ResolvedTrack.raw) for
  it. ChunkAssembler and mux_chunk already handle fmp4 tracks.

Contract for every source: next_group(timeout) returns within `timeout` seconds
plus at most one request's own deadline, whatever the network does (no retry
waits beyond the timeout), and aclose() may be called at any time and returns
promptly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from .capture import FORMAT_SORT
from .config import Config
from .models import MediaType
from .youtube_timeline import (
    TRACK_AUDIO,
    TRACK_MUXED,
    TRACK_VIDEO,
    TS_WRAP_SECONDS,
    SegmentGroup,
    TrackSegment,
    inspect_segment,
)
from .ytdlp import auth_args

logger = logging.getLogger(__name__)

# Measured: VOD time = HLS clock + 0.036 s, constant across a 3.3 h stream and a stream break.
HLS_CLOCK_TO_VOD_SECONDS = 0.036


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class SourceError(Exception):
    """A transient failure that outlasted the source's own retries (network
    errors, server errors). The capture backs off and, if it persists, reconnects."""


class SourceExpiredError(SourceError):
    """The URLs were rejected (401/403/410, playlist gone): re-resolve them with
    yt-dlp and continue from the same sequence."""


class SourceResetError(SourceError):
    """The stream's sequence numbering restarted: the resume point and the clock
    state no longer apply. Start again at the live edge."""


class ResolveError(Exception):
    """yt-dlp could not resolve the stream right now (retryable)."""


class ResolveStopped(ResolveError):
    """should_stop() turned true while yt-dlp was running."""


class StreamNotLiveError(ResolveError):
    """The stream is no longer (or not yet) live: end the run."""


class StreamUnavailableError(ResolveError):
    """The stream can't be watched with this account (members-only, private): end the run."""


class UnsupportedSourceError(ResolveError):
    """yt-dlp offered no format this worker can read (e.g. SABR-only)."""


# ----------------------------------------------------------------------
# Resolution (yt-dlp)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedTrack:
    track: str  # TRACK_VIDEO / TRACK_AUDIO / TRACK_MUXED
    protocol: str
    url: str
    format_id: str
    http_headers: dict[str, str] = field(default_factory=dict)
    raw: dict = field(default_factory=dict)  # yt-dlp's format dict, for transports that need more than the URL


@dataclass(frozen=True)
class ResolvedStream:
    video_id: str
    live_status: str
    tracks: tuple[ResolvedTrack, ...]
    expires_at: float | None

    def track(self, name: str) -> ResolvedTrack | None:
        return next((t for t in self.tracks if t.track == name), None)


@dataclass
class SourceContext:
    """What a source may need besides the resolved formats."""

    key: str
    watch_url: str
    media_type: MediaType
    should_stop: Callable[[], bool]
    log: Callable[[str], None]  # event sink: app log + per-channel run log
    max_lag_seconds: float = 600.0  # catch up from at most this far behind the live edge


_YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HLS_PROTOCOLS = ("m3u8", "m3u8_native")
# Reasons yt-dlp may come back without formats that a later attempt can fix.
_TRANSIENT_HINTS = (
    "po token",
    "potoken",
    "n challenge",
    "nsig",
    "signature",
    "not a bot",
    "try again later",
    "rate-limit",
    "rate limit",
    "too many requests",
    "http error 429",
)
_UNAVAILABLE_HINTS = (
    "members-only",
    "members only",
    "join this channel",
    "private video",
    "video unavailable",
    "sign in to confirm your age",
)
_NOT_LIVE_HINTS = ("not currently live", "has ended", "this live event will begin", "premieres in", "is offline")
# live_status values that definitely mean "not live now".
_NOT_LIVE_STATUSES = ("was_live", "post_live", "not_live", "is_upcoming")


def watch_url(url: str, stream_id: str) -> str:
    """The stream's own watch URL. A channel /live URL can point at a different
    broadcast by the time the capture re-resolves, so pin the video id."""
    if stream_id and _YOUTUBE_ID.match(stream_id):
        return f"https://www.youtube.com/watch?v={stream_id}"
    return url


def format_args(media_type: MediaType) -> list[str]:
    """Format selection restricted to HLS. The protocol filter is safe (every
    format reports its protocol); codec preferences stay a sort order, as in the
    live-edge capture (capture.FORMAT_SORT), because YouTube reports the HLS
    audio codec as unknown. Media types that don't send video don't download it."""
    if media_type is MediaType.VIDEO:
        selector = "bv[protocol^=m3u8]+ba[protocol^=m3u8]/b[protocol^=m3u8]"
    else:
        selector = "ba[protocol^=m3u8]/b[protocol^=m3u8]"
    return ["-f", selector, "-S", FORMAT_SORT]


def build_resolve_command(config: Config, url: str, media_type: MediaType) -> list[str]:
    # --ignore-no-formats-error: when no HLS format matches, yt-dlp still prints
    # the JSON, so the reason (ended, SABR-only, PO token) can be told apart.
    return [
        config.capture.yt_dlp_path,
        "-j",
        "--no-playlist",
        "--ignore-no-formats-error",
        *auth_args(config, url, "download"),
        *format_args(media_type),
        url,
    ]


def _url_expiry(url: str) -> float | None:
    m = re.search(r"/expire/(\d+)", url)
    if m:
        return float(m.group(1))
    values = parse_qs(urlparse(url).query).get("expire")
    if values and values[0].isdigit():
        return float(values[0])
    return None


def _classify(fmt: dict) -> str:
    vcodec = fmt.get("vcodec")
    acodec = fmt.get("acodec")
    if vcodec == "none":
        return TRACK_AUDIO
    if acodec == "none":
        return TRACK_VIDEO
    if vcodec in (None, "") and acodec not in (None, "", "none"):
        return TRACK_AUDIO
    return TRACK_MUXED


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_resolved(metadata: dict, stderr: str = "") -> ResolvedStream:
    """Builds a ResolvedStream from yt-dlp's -j output. With
    --ignore-no-formats-error, YouTube's playability errors (bot checks, rate
    limits, members-only) arrive here as warnings next to a JSON without
    formats, so the reason is read from both. Raises StreamNotLiveError,
    StreamUnavailableError, UnsupportedSourceError or ResolveError (retryable)."""
    lowered = stderr.lower()
    live_status = metadata.get("live_status") or ("is_live" if metadata.get("is_live") else None)
    if live_status in _NOT_LIVE_STATUSES:
        raise StreamNotLiveError(f"live_status is {live_status}")
    if any(hint in lowered for hint in _UNAVAILABLE_HINTS):
        raise StreamUnavailableError(_last_line(stderr))
    if live_status != "is_live":
        # YouTube didn't say (bot check, rate limit, an error page): ask again later.
        raise ResolveError(f"yt-dlp reported no live status ({_last_line(stderr) or 'no reason given'})")

    selected = metadata.get("requested_formats") or ([metadata] if metadata.get("url") else [])
    if not selected:
        formats = metadata.get("formats") or []
        protocols = sorted({str(f.get("protocol") or "") for f in formats})
        if any("sabr" in p for p in protocols):
            raise UnsupportedSourceError(
                "YouTube only offers this stream over SABR, which this worker can't read yet (add a SABR SegmentSource)"
            )
        if not formats or any(hint in lowered for hint in _TRANSIENT_HINTS):
            raise ResolveError(f"yt-dlp got no usable formats ({_last_line(stderr) or 'no reason given'})")
        raise UnsupportedSourceError(f"no HLS format offered (protocols: {', '.join(protocols)})")

    tracks: list[ResolvedTrack] = []
    for fmt in selected:
        protocol = str(fmt.get("protocol") or "")
        format_id = str(fmt.get("format_id") or "?")
        if protocol not in _HLS_PROTOCOLS:
            if "sabr" in protocol:
                raise UnsupportedSourceError(f"format {format_id} is only available over SABR, which this worker can't read yet")
            raise UnsupportedSourceError(f"format {format_id} uses protocol '{protocol}', not HLS")
        url = fmt.get("url")
        if not url:
            raise UnsupportedSourceError(f"format {format_id} has no URL")
        headers = {str(k): str(v) for k, v in (fmt.get("http_headers") or metadata.get("http_headers") or {}).items()}
        tracks.append(ResolvedTrack(_classify(fmt), protocol, url, format_id, headers, dict(fmt)))

    names = [t.track for t in tracks]
    if len(set(names)) != len(names):
        raise UnsupportedSourceError(f"ambiguous format selection: {names}")
    if TRACK_AUDIO not in names and TRACK_MUXED not in names:
        raise UnsupportedSourceError("no audio in the selected formats")

    expiries = [e for e in (_url_expiry(t.url) for t in tracks) if e is not None]
    return ResolvedStream(
        video_id=str(metadata.get("id") or ""),
        live_status=live_status,
        tracks=tuple(tracks),
        expires_at=min(expiries) if expiries else None,
    )


def classify_resolve(returncode: int, stdout: str, stderr: str) -> ResolvedStream:
    """Interprets a finished `yt-dlp -j` run."""
    lowered = stderr.lower()
    if returncode != 0:
        last = _last_line(stderr) or f"exit {returncode}"
        if any(hint in lowered for hint in _NOT_LIVE_HINTS):
            raise StreamNotLiveError(last)
        if any(hint in lowered for hint in _UNAVAILABLE_HINTS):
            raise StreamUnavailableError(last)
        raise ResolveError(last)
    if not stdout.strip():
        # A --match-filter (members-only) rejected the video: nothing to resolve.
        raise StreamUnavailableError(_last_line(stderr) or "yt-dlp skipped the video (members-only?)")
    try:
        metadata = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ResolveError(f"invalid JSON from yt-dlp: {e}") from e
    if not isinstance(metadata, dict):
        raise ResolveError("unexpected JSON from yt-dlp")
    return parse_resolved(metadata, stderr)


async def _kill_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    if sys.platform != "win32":
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=5)


async def resolve_stream(
    config: Config,
    url: str,
    media_type: MediaType,
    should_stop: Callable[[], bool],
    raw_log: Callable[[str], None] | None = None,
    timeout: float = 90.0,
) -> ResolvedStream:
    """Runs `yt-dlp -j` for the stream and returns its HLS tracks. Polls
    should_stop while yt-dlp runs. yt-dlp's stderr goes to raw_log.
    Raises ResolveStopped, StreamNotLiveError, StreamUnavailableError,
    UnsupportedSourceError or ResolveError.

    A yt-dlp still running is killed with its whole process group on stop, on
    timeout and when the caller is cancelled. Its output goes to temporary
    files rather than pipes: the result waits only for yt-dlp itself, never
    for a child still holding a pipe open, and a large -j never stalls on a
    full pipe."""
    cmd = build_resolve_command(config, url, media_type)
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            # Own session: yt-dlp's children (deno) die with it when it's killed.
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=sys.platform != "win32",
            )
        except OSError as e:
            raise ResolveError(f"yt-dlp could not be started: {e}") from e
        deadline = time.monotonic() + timeout
        exited = asyncio.ensure_future(proc.wait())
        try:
            while not exited.done():
                if should_stop():
                    raise ResolveStopped("stopped while resolving")
                if time.monotonic() > deadline:
                    raise ResolveError(f"yt-dlp timed out after {timeout:.0f}s")
                await asyncio.wait({exited}, timeout=0.2)
        finally:
            exited.cancel()
            await _kill_process(proc)
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode("utf-8", errors="replace")
        stderr = err.read().decode("utf-8", errors="replace")
    if raw_log is not None and stderr.strip():
        with contextlib.suppress(OSError, ValueError):
            raw_log(stderr.rstrip() + "\n")
    return classify_resolve(proc.returncode if proc.returncode is not None else 1, stdout, stderr)


# ----------------------------------------------------------------------
# Source interface
# ----------------------------------------------------------------------


class SegmentSource(ABC):
    """Delivers SegmentGroups in sequence order.

    `resume` is the first sequence the capture still needs (None: the live edge).
    A source never delivers a sequence before it; when it can't serve it (too
    far behind, no longer available) it starts at the live edge or the oldest
    available sequence instead and says so in the log. `checkpoint` is the
    first sequence not yet delivered.

    Clock semantics: segment clocks are in the source's own domain;
    `wrap_period` is how often that clock wraps (None: never) and `vod_offset`
    what to add to it to get VOD time.
    """

    wrap_period: float | None = None
    vod_offset: float = 0.0

    # Time source (overridable in tests).
    _monotonic = staticmethod(time.monotonic)

    def __init__(self, resolved: ResolvedStream, resume: int | None, ctx: SourceContext):
        self.resolved = resolved
        self.ctx = ctx
        self._next_sequence = resume
        self._ended = False
        # Counters for the run summary and the capture's health checks.
        self.segments_missing = 0
        self.segments_skipped = 0
        # Per track: sequences in a row whose segment couldn't be fetched or read.
        self.consecutive_missing: dict[str, int] = {}

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(t.track for t in self.resolved.tracks)

    @property
    def clock_track(self) -> str:
        return TRACK_AUDIO if TRACK_AUDIO in self.tracks else TRACK_MUXED

    @property
    def checkpoint(self) -> int | None:
        return self._next_sequence

    @property
    def ended(self) -> bool:
        """True once the source has seen the end of the stream."""
        return self._ended

    @property
    def expires_at(self) -> float | None:
        """When the capture must have re-resolved the URLs, or None when the source
        takes care of that itself."""
        return self.resolved.expires_at

    @property
    def behind_live(self) -> float | None:
        """Seconds of stream the source has seen published past what it delivered
        (0 at the live edge), or None when it can't tell yet."""
        return None

    def log(self, message: str) -> None:
        self.ctx.log(message)

    @abstractmethod
    async def next_group(self, timeout: float) -> SegmentGroup | None:
        """The next sequence's segments, or None if it isn't available within
        about `timeout` seconds (or the stream ended: check `ended`). Raises
        SourceExpiredError, SourceResetError or SourceError."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Stops everything the source runs. Idempotent, prompt, never raises."""

    async def _sleep(self, seconds: float) -> None:
        end = self._monotonic() + seconds
        while not self.ctx.should_stop():
            remaining = end - self._monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.2, remaining))


def create_segment_source(resolved: ResolvedStream, resume: int | None, ctx: SourceContext) -> SegmentSource:
    """The source for the resolved transport. Add SABR here."""
    if all(t.protocol in _HLS_PROTOCOLS for t in resolved.tracks):
        return HlsSegmentSource(resolved, resume, ctx)
    raise UnsupportedSourceError(f"no segment source for protocols {sorted({t.protocol for t in resolved.tracks})}")


# ----------------------------------------------------------------------
# HLS
# ----------------------------------------------------------------------


@dataclass
class MediaPlaylist:
    media_sequence: int
    target_duration: float
    segments: list[tuple[int, float, str]]  # (sequence, duration, absolute URL)
    ended: bool

    @property
    def last_sequence(self) -> int | None:
        return self.segments[-1][0] if self.segments else None

    @property
    def first_sequence(self) -> int | None:
        return self.segments[0][0] if self.segments else None


def parse_media_playlist(text: str, base_url: str) -> MediaPlaylist | None:
    """Parses an HLS media playlist. None if it isn't one (e.g. a master playlist)."""
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0] != "#EXTM3U" or any(line.startswith("#EXT-X-STREAM-INF") for line in lines):
        return None
    media_sequence = 0
    target_duration = 0.0
    ended = False
    segments: list[tuple[int, float, str]] = []
    pending_duration: float | None = None
    for line in lines[1:]:
        if not line:
            continue
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            try:
                media_sequence = int(line.split(":", 1)[1])
            except ValueError:
                return None
        elif line.startswith("#EXT-X-TARGETDURATION:"):
            try:
                target_duration = float(line.split(":", 1)[1])
            except ValueError:
                return None
        elif line.startswith("#EXTINF:"):
            try:
                pending_duration = float(line.split(":", 1)[1].split(",", 1)[0])
            except ValueError:
                pending_duration = None
        elif line == "#EXT-X-ENDLIST":
            ended = True
        elif not line.startswith("#"):
            sequence = media_sequence + len(segments)
            segments.append((sequence, pending_duration if pending_duration is not None else target_duration, urljoin(base_url, line)))
            pending_duration = None
    return MediaPlaylist(media_sequence=media_sequence, target_duration=target_duration, segments=segments, ended=ended)


_SQ_IN_URL = re.compile(r"/sq/(\d+)(?=/|$)")


def sequence_url_template(segment_url: str, sequence: int) -> str | None:
    """A str.format template ('…/sq/{}/…') for fetching any sequence number
    from a YouTube segment URL, or None when the URL doesn't carry it. YouTube
    serves every segment of the current stream period by number, including ones
    that already left the (tiny) live playlist window.

    The rest of the URL isn't fixed for the whole stream: YouTube can change
    parameters mid-stream (e.g. `lmt` after an encoder change), and a URL with a
    stale value is a 404. So templates come from the newest segment listed."""
    m = _SQ_IN_URL.search(segment_url)
    if not m or int(m.group(1)) != sequence:
        return None
    escaped = segment_url.replace("{", "{{").replace("}", "}}")
    m = _SQ_IN_URL.search(escaped)
    assert m is not None
    return f"{escaped[: m.start(1)]}{{}}{escaped[m.end(1) :]}"


def url_changes(old: str, new: str, limit: int = 3) -> str:
    """What differs between two segment URL templates, for the log, e.g.
    'goap: lmt%3D33 -> lmt%3D34' (YouTube paths are /key/value pairs)."""
    old_parts, new_parts = old.split("/"), new.split("/")
    if len(old_parts) != len(new_parts):
        return "a different path layout"
    changes = [
        f"{old_parts[i - 1] + ': ' if i else ''}{old_parts[i][:40]} -> {new_parts[i][:40]}"
        for i in range(len(old_parts))
        if old_parts[i] != new_parts[i]
    ]
    more = f" and {len(changes) - limit} more" if len(changes) > limit else ""
    return ", ".join(changes[:limit]) + more


@dataclass
class _HlsTrack:
    resolved: ResolvedTrack
    playlist: MediaPlaylist | None = None
    template: str | None = None
    urls: dict[int, tuple[float, str]] = field(default_factory=dict)  # sequence -> (duration, url)
    last_poll: float = -1e9
    retry_after: float = -1e9  # a secondary playlist that failed isn't polled again before this
    not_before: float = -1e9  # a playlist whose retry was cut short isn't requested again before this
    # The template may no longer match YouTube's current URLs (the clock track's URLs
    # changed, or a URL built from it got a 404): re-read the playlist before using it.
    stale: bool = False

    @property
    def head(self) -> int | None:
        return self.playlist.last_sequence if self.playlist else None

    def url_for(self, sequence: int) -> tuple[float | None, str | None]:
        entry = self.urls.get(sequence)
        if entry is not None:
            return entry
        if self.template is not None:
            return None, self.template.format(sequence)
        return None, None


@dataclass
class _Progress:
    """How far a segment download got, for the log when it's too late. Written by
    the download task, read by the source."""

    started: float | None = None  # monotonic; None until the download task first runs
    responded: bool = False  # the current attempt got response headers
    received: int = 0  # body bytes of the current attempt
    error: str = ""  # the last failed response


class _Closed(Exception):
    """The source was closed, should_stop() turned true, or this fetch was abandoned (cancelled)."""


class _AttemptFailed(Exception):
    """One HTTP attempt failed in a retryable way."""


class _OutOfTime(Exception):
    """The caller's time budget ran out before the request could be (re)tried."""


@dataclass
class _Fetched:
    segment: TrackSegment | None
    reason: str = ""  # why segment is None: "not available (404)", "unreadable (...)", "...: HTTP 403", ...
    is_404: bool = False


def _complete_length(resp: httpx.Response) -> int | None:
    """The body's length on the wire when a completed body can be recognised
    by it (Content-Length without a content encoding), else None."""
    if resp.headers.get("Content-Encoding", "identity").strip().lower() not in ("", "identity"):
        return None
    try:
        return int(resp.headers["Content-Length"])
    except (KeyError, ValueError):
        return None


class HlsSegmentSource(SegmentSource):
    """Follows YouTube's HLS media playlists and fetches each sequence's segments
    directly (no ffmpeg HLS reader), so every segment's clock is read from its
    own bytes and video and audio are paired by sequence number.

    Only the clock track's playlist is polled for the live edge; the other
    tracks' segments are fetched by number. Every download is its own task, up
    to PREFETCH_SEQUENCES sequences ahead per track, which lets the source catch
    up after a stall (VOD-accurate lines don't mind arriving late) as long as it
    isn't more than ctx.max_lag_seconds behind, past which it jumps to the live
    edge. A video segment that doesn't arrive in time is left out (gap in the
    video only) and its download cancelled: audio is never held up by video, and
    a broken video playlist never stops audio.
    """

    wrap_period = TS_WRAP_SECONDS
    vod_offset = HLS_CLOCK_TO_VOD_SECONDS

    PREFETCH_SEQUENCES = 4
    CONNECT_TIMEOUT = 3.0
    READ_TIMEOUT = 5.0
    REQUEST_ATTEMPTS = 3
    # Clock-track downloads failing outright this many times in a row ⇒ SourceError.
    MAX_CONSECUTIVE_FETCH_FAILURES = 6
    # How long other tracks may lag behind the clock track's segment, in target durations.
    VIDEO_GRACE_TARGETS = 2.0
    # A 404 on a segment at the live edge is retried this long (it may not be on this CDN node yet).
    EDGE_404_WAIT_TARGETS = 3.0
    # A playlist far behind the next sequence must stay so for this many polls and this long
    # (in target durations, at least RESET_MIN_SECONDS) before a numbering restart is suspected...
    RESET_POLLS = 3
    RESET_TARGETS = 3.0
    RESET_MIN_SECONDS = 10.0
    # ...and is only believed when the last delivered sequence can no longer be fetched by number.
    # 404s in a row while fetching older segments by number ⇒ they're gone (stream break).
    BACKFILL_404_LIMIT = 3
    # A failing secondary playlist (only polled without a URL template) is retried this rarely.
    SECONDARY_POLL_INTERVAL = 10.0
    DEFAULT_TARGET = 2.0

    def __init__(
        self,
        resolved: ResolvedStream,
        resume: int | None,
        ctx: SourceContext,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ):
        super().__init__(resolved, resume, ctx)
        self._tracks: dict[str, _HlsTrack] = {t.track: _HlsTrack(resolved=t) for t in resolved.tracks}
        headers = next((t.http_headers for t in resolved.tracks if t.http_headers), {})
        self._client_factory = client_factory or (lambda: self._new_client(headers))
        self._client: httpx.AsyncClient | None = None  # built on first use, closed by aclose()
        # Every fetch is its own task: audio downloads never queue behind stalled video
        # ones, and a download dropped as too late is cancelled at once (a thread pool
        # had to leave it running until its response started).
        self._tasks: dict[tuple[str, int], asyncio.Future[_Fetched]] = {}
        self._running: set[asyncio.Future[_Fetched]] = set()  # unfinished fetch tasks, dropped ones included
        self._closed = False
        self._opened = False
        self._target = self.DEFAULT_TARGET
        self._fetch_failures = 0
        self._stale_polls = 0
        self._stale_since: float | None = None
        self._backfill_404s = 0
        self._backfill_counted: int | None = None  # the sequence last counted as a backfill 404
        self._done_at: dict[tuple[str, int], float] = {}  # when each fetch finished
        self._progress: dict[tuple[str, int], _Progress] = {}  # how far each fetch got, for the log
        self._missing_since: dict[str, tuple[int, str]] = {}  # track -> (first missing sequence, reason)

    # -- lifecycle ----------------------------------------------------------

    async def aclose(self) -> None:
        self._closed = True
        tasks = list(self._running)
        for task in tasks:
            task.cancel()
        self._tasks.clear()
        try:
            if tasks:
                await asyncio.wait(tasks)
        finally:
            for name in list(self._missing_since):
                self._end_missing_streak(name, None)
            client, self._client = self._client, None
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.aclose()

    def _stopping(self) -> bool:
        return self._closed or self.ctx.should_stop()

    async def _sleep(self, seconds: float) -> None:
        end = self._monotonic() + seconds
        while not self._stopping():
            remaining = end - self._monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.2, remaining))

    def _new_client(self, headers: dict[str, str]) -> httpx.AsyncClient:
        # googlevideo answers with redirects: requests followed them by default, httpx doesn't.
        timeout = httpx.Timeout(connect=self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT, write=self.READ_TIMEOUT, pool=self.READ_TIMEOUT)
        return httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout)

    def _http(self) -> httpx.AsyncClient:
        if self._closed:
            raise _Closed()
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    @property
    def _poll_interval(self) -> float:
        return max(0.5, min(5.0, self._target))

    @property
    def behind_live(self) -> float | None:
        """The clock track's listed sequences not delivered yet, in target durations
        (as the lag check counts them), plus the time since its playlist was read:
        the playlist isn't re-read while listed sequences remain, and the live edge
        moves on meanwhile."""
        state = self._tracks[self.clock_track]
        head = state.head
        if head is None or self._next_sequence is None:
            return None
        return max(0.0, (head - self._next_sequence + 1) * self._target) + max(0.0, self._monotonic() - state.last_poll)

    # -- HTTP -----------------------------------------------------------------

    async def _read_body(self, resp: httpx.Response, deadline: float, deadline_seconds: float, progress: _Progress | None = None) -> bytes:
        """Reads a response body piece by piece as data arrives, checking stop
        and the wall-clock deadline after every piece, so a trickling response
        can't hold the source longer than the deadline (each piece is bounded by
        READ_TIMEOUT). A body that completes is never discarded."""
        length = _complete_length(resp)
        parts: list[bytes] = []
        async for piece in resp.aiter_bytes():
            parts.append(piece)
            if progress is not None:
                progress.received += len(piece)
            if length is not None and resp.num_bytes_downloaded >= length:
                # Complete. The iteration ends without another network read; finishing it
                # (rather than breaking out) lets the connection go back to the pool.
                continue
            if self._stopping():
                raise _Closed()
            if self._monotonic() > deadline:
                raise _AttemptFailed(f"download slower than {deadline_seconds:.0f}s")
        return b"".join(parts)

    async def _get_once(self, url: str, deadline_seconds: float, progress: _Progress | None = None) -> tuple[int, bytes, float | None]:
        """One GET with its body streamed under a wall-clock deadline. Returns
        (status, body, retry_after)."""
        deadline = self._monotonic() + deadline_seconds
        if progress is not None:
            progress.responded, progress.received = False, 0
        try:
            async with self._http().stream("GET", url) as resp:
                if progress is not None:
                    progress.responded = True
                status = resp.status_code
                if status != 200:
                    retry_after = None
                    header = resp.headers.get("Retry-After")
                    if header is not None:
                        with contextlib.suppress(TypeError, ValueError):
                            retry_after = float(header)
                    return status, b"", retry_after
                return 200, await self._read_body(resp, deadline, deadline_seconds, progress), None
        except (httpx.HTTPError, httpx.InvalidURL, OSError) as e:
            raise _AttemptFailed(f"{type(e).__name__}: {e}") from e

    async def _download(
        self,
        url: str,
        what: str,
        deadline_seconds: float,
        edge_wait: float = 0.0,
        until: float | None = None,
        progress: _Progress | None = None,
    ) -> bytes | None:
        """GET with retries. Returns the body, or None for a 404 (after waiting up
        to edge_wait for a segment at the live edge, else after one retry) or
        another permanent client error. Raises SourceExpiredError on 401/403/410,
        SourceError once transient failures outlast the retries, _Closed when
        stopped (an abandoned download is cancelled instead), and _OutOfTime when
        a retry wait would run past `until` (the caller's budget; an attempt that
        started runs to its own deadline). `progress` follows the download, for the log."""
        last_error = "no attempt"
        failures = 0
        not_found = 0
        edge_until = self._monotonic() + edge_wait

        async def pause(seconds: float) -> None:
            if until is not None and self._monotonic() + seconds > until:
                raise _OutOfTime()
            await self._sleep(seconds)

        while failures < self.REQUEST_ATTEMPTS:
            if self._stopping():
                raise _Closed()
            try:
                status, body, retry_after = await self._get_once(url, deadline_seconds, progress)
            except _AttemptFailed as e:
                failures += 1
                last_error = str(e)
                if progress is not None:
                    progress.error = last_error
                await pause(min(4.0, 0.5 * 2 ** (failures - 1)))
                continue
            if status == 200:
                return body
            if progress is not None:
                progress.error = f"HTTP {status}"
            if status in (401, 403, 410):
                raise SourceExpiredError(f"{what}: HTTP {status}")
            if status == 404:
                if self._monotonic() < edge_until:
                    await pause(max(0.25, self._target / 2))
                    continue
                not_found += 1
                if not_found < 2:
                    await pause(0.5)
                    continue
                return None
            if status == 429 or status >= 500:
                failures += 1
                last_error = f"HTTP {status}"
                wait = retry_after if status == 429 and retry_after else 0.5 * 2**failures
                await pause(min(30.0, wait))
                continue
            self.log(f"{what}: HTTP {status}, giving up on it")
            return None
        raise SourceError(f"{what}: {last_error}")

    # -- playlists ----------------------------------------------------------

    async def _poll(self, name: str, until: float | None = None) -> None:
        state = self._tracks[name]
        state.last_poll = self._monotonic()
        try:
            body = await self._download(state.resolved.url, f"{name} playlist", max(4.0, 2 * self._target), until=until)
        except _OutOfTime:
            # Retried on a later call, not before the wait that was cut short.
            state.not_before = self._monotonic() + self._poll_interval
            raise
        if body is None:
            # A live playlist that 404s is gone: let yt-dlp tell whether the stream is still live.
            raise SourceExpiredError(f"{name} playlist is gone")
        playlist = parse_media_playlist(body.decode("utf-8", errors="replace"), state.resolved.url)
        if playlist is None:
            raise SourceError(f"{name} playlist is not an HLS media playlist")
        state.playlist = playlist
        if playlist.target_duration > 0:
            self._target = min(10.0, max(0.5, playlist.target_duration))
        for sequence, duration, url in playlist.segments:
            state.urls[sequence] = (duration, url)
        if playlist.segments:
            newest_sequence, _, newest_url = playlist.segments[-1]
            template = sequence_url_template(newest_url, newest_sequence)
            if template is not None:
                if state.template is not None and template != state.template and name == self.clock_track:
                    # YouTube changed its segment URLs (a new lmt, ...): the other tracks'
                    # templates are out of date too.
                    self.log(
                        f"segment URLs changed at sequence {newest_sequence} ({url_changes(state.template, template)}); "
                        "refreshing the other tracks' URLs"
                    )
                    for other, other_state in self._tracks.items():
                        if other != name:
                            other_state.stale = True
                state.template = template
            state.stale = False
        if playlist.first_sequence is not None:
            for old in [s for s in state.urls if s < playlist.first_sequence - 600]:
                del state.urls[old]
        if name == self.clock_track:
            self._note_staleness(playlist)

    async def _poll_secondary(self, name: str, until: float | None = None, min_interval: float = 0.0) -> bool:
        """Polls a non-clock track's playlist without letting its failure reach
        the caller (audio must not depend on it). After a failure it isn't tried
        again for SECONDARY_POLL_INTERVAL seconds, and never within min_interval of
        the previous poll. True if it was polled."""
        state = self._tracks[name]
        now = self._monotonic()
        if now < max(state.retry_after, state.not_before) or now - state.last_poll < min_interval:
            return False
        try:
            await self._poll(name, until)
            return True
        except _OutOfTime:
            return False
        except SourceError as e:
            state.retry_after = self._monotonic() + self.SECONDARY_POLL_INTERVAL
            self.log(f"{name} playlist unavailable ({e}); continuing without it")
            return False

    async def _open(self, until: float) -> None:
        # The clock track's playlist decides the live edge; the others are polled once for their URL template.
        await self._poll(self.clock_track, until)
        for name in self._tracks:
            if name != self.clock_track:
                await self._poll_secondary(name, until)
        clock_head = self._tracks[self.clock_track].head
        if clock_head is None:
            raise SourceError("the playlist lists no segments yet")
        if self._next_sequence is None:
            heads = [t.head for t in self._tracks.values() if t.head is not None and t.head >= clock_head - 2]
            self._next_sequence = min(heads) if heads else clock_head
            self.log(f"starting at the live edge, sequence {self._next_sequence}")
        else:
            self.log(f"resuming at sequence {self._next_sequence} (live edge {clock_head})")
        self._opened = True

    def _note_staleness(self, playlist: MediaPlaylist) -> None:
        """Counts distinct polls whose head is far below the next sequence."""
        if self._next_sequence is None or playlist.last_sequence is None:
            return
        window = max(3, len(playlist.segments))
        if playlist.last_sequence < self._next_sequence - window:
            self._stale_polls += 1
            if self._stale_since is None:
                self._stale_since = self._monotonic()
        else:
            self._stale_polls = 0
            self._stale_since = None

    async def _check_reset(self, until: float) -> None:
        """Raises SourceResetError once the numbering has demonstrably restarted:
        the playlist stayed far behind for a while, and the last delivered
        sequence is no longer served by number. A merely stale playlist (an edge
        node lagging after a reconnect) passes the second test."""
        if self._stale_since is None or self._stale_polls < self.RESET_POLLS:
            return
        if self._monotonic() - self._stale_since < max(self.RESET_MIN_SECONDS, self.RESET_TARGETS * self._target):
            return
        assert self._next_sequence is not None
        state = self._tracks[self.clock_track]
        _, url = state.url_for(self._next_sequence - 1)
        head = state.head
        if url is not None:
            try:
                still_there = await self._download(
                    url, f"{self.clock_track} sequence {self._next_sequence - 1}", max(4.0, 2 * self._target), until=until
                )
            except SourceExpiredError:
                still_there = None
            except SourceError:
                return  # can't tell right now; check again on the next poll
            if still_there is not None:
                self.log(f"the playlist is stale (ends at {head}, expected {self._next_sequence}); waiting for it to catch up")
                self._stale_polls = 0
                self._stale_since = self._monotonic()
                return
        raise SourceResetError(f"sequence numbers went back from {self._next_sequence} to {head}")

    def _jump(self, sequence: int, reason: str) -> None:
        assert self._next_sequence is not None
        if sequence > self._next_sequence:
            self.segments_skipped += sequence - self._next_sequence
        self.log(f"{reason}; continuing at sequence {sequence}")
        for key in [k for k in self._tasks if k[1] < sequence]:
            self._drop(key)
        self._next_sequence = sequence

    def _drop(self, key: tuple[str, int]) -> None:
        """Forgets a fetch and cancels its download (a finished one ignores that)."""
        task = self._tasks.pop(key, None)
        self._done_at.pop(key, None)
        self._progress.pop(key, None)
        if task is not None:
            task.cancel()

    # -- segments -----------------------------------------------------------

    async def next_group(self, timeout: float) -> SegmentGroup | None:
        try:
            return await self._next_group(timeout)
        except (_Closed, _OutOfTime):
            return None

    async def _next_group(self, timeout: float) -> SegmentGroup | None:
        deadline = self._monotonic() + timeout
        if not self._opened:
            await self._open(deadline)
        assert self._next_sequence is not None
        clock_state = self._tracks[self.clock_track]

        # 1. Wait for the clock track to list the next sequence.
        while clock_state.head is None or clock_state.head < self._next_sequence:
            if clock_state.playlist is not None and clock_state.playlist.ended:
                self._ended = True
                return None
            await self._check_reset(deadline)
            now = self._monotonic()
            if now >= deadline:
                return None
            next_poll = max(clock_state.last_poll + self._poll_interval, clock_state.not_before)
            if next_poll > now:
                await self._sleep(min(next_poll - now, deadline - now))
                if self._stopping() or self._monotonic() < next_poll:
                    return None
            await self._poll(self.clock_track, deadline)
        head = clock_state.head
        assert head is not None

        # 2. Too far behind: back to the live edge.
        lag = (head - self._next_sequence) * self._target
        if lag > self.ctx.max_lag_seconds:
            self._jump(head, f"{lag:.0f}s behind the live edge, skipping {head - self._next_sequence} segments")

        # 3. Keep a few sequences in flight.
        sequence = self._next_sequence
        for s in range(sequence, min(head, sequence + self.PREFETCH_SEQUENCES - 1) + 1):
            for name in self._tracks:
                await self._submit(name, s, head, deadline)

        # 4. The clock track's segment.
        clock_task = self._tasks[(self.clock_track, sequence)]
        if not await self._wait(clock_task, deadline):
            return None
        clock_result = self._result(self.clock_track, sequence, clock_task)
        if clock_result is None:
            # Transient failure, not escalated: the sequence is retried on the next call.
            return None
        window_first = clock_state.playlist.first_sequence if clock_state.playlist else None
        if clock_result.is_404 and window_first is not None and sequence < window_first:
            if self._backfill_counted != sequence:  # count each sequence once, however often it's revisited
                self._backfill_counted = sequence
                self._backfill_404s += 1
            if self._backfill_404s >= self.BACKFILL_404_LIMIT:
                self._backfill_404s = 0
                self._jump(window_first, f"segments before {sequence} are no longer available (stream break?)")
                return None
        else:
            self._backfill_404s = 0

        # 5. The other tracks, given a short grace period from when this sequence's
        # clock segment arrived (prefetched segments have usually used it up already,
        # so persistently late video can't slow the clock track down).
        clock_done = self._done_at.get((self.clock_track, sequence), self._monotonic())
        grace_end = clock_done + max(1.0, self.VIDEO_GRACE_TARGETS * self._target)
        for name in self._tracks:
            if name == self.clock_track:
                continue
            task = self._tasks[(name, sequence)]
            if not task.done() and not await self._wait(task, min(deadline, grace_end)) and self._monotonic() < grace_end:
                return None  # still within the grace period: try again on the next call

        # 6. Assemble the group.
        group = SegmentGroup(sequence=sequence)
        for name in self._tracks:
            if name == self.clock_track:
                fetched = clock_result
            else:
                task = self._tasks[(name, sequence)]
                fetched = (self._result(name, sequence, task) if task.done() else None) or self._too_late(name, sequence)
            self._record(name, sequence, fetched, group)
        for name in self._tracks:
            self._drop((name, sequence))
        self._next_sequence = sequence + 1
        return group

    async def _submit(self, name: str, sequence: int, head: int, until: float | None = None) -> None:
        key = (name, sequence)
        if key in self._tasks:
            return
        state = self._tracks[name]
        nominal, url = state.url_for(sequence)
        if name != self.clock_track and sequence not in state.urls:
            if url is None:
                # No template for this track: its playlist may list the sequence.
                refreshed = await self._poll_secondary(name, until)
            elif state.stale:
                # The template may be out of date: re-read the playlist (at most once per segment).
                refreshed = await self._poll_secondary(name, until, min_interval=max(1.0, self._target))
            else:
                refreshed = False
            if refreshed:
                nominal, url = state.url_for(sequence)
        task: asyncio.Future[_Fetched]
        if url is None:
            task = asyncio.get_running_loop().create_future()
            task.set_result(_Fetched(None, "no URL for it"))
        else:
            edge_wait = self.EDGE_404_WAIT_TARGETS * self._target if sequence >= head - 1 else 0.0
            progress = self._progress[key] = _Progress()
            task = asyncio.create_task(
                self._fetch(name, sequence, url, nominal or self._target, edge_wait, progress),
                name=f"timeline-{self.ctx.key}-{name}-{sequence}",
            )
            self._running.add(task)
        self._tasks[key] = task
        task.add_done_callback(lambda done, key=key: self._finished(key, done))

    def _finished(self, key: tuple[str, int], task: asyncio.Future[_Fetched]) -> None:
        """A fetch task's done callback: stamps when the fetch finished (unless it
        was dropped meanwhile) and marks its outcome as seen, so an abandoned
        fetch's error is never reported as 'never retrieved'."""
        self._running.discard(task)
        if not task.cancelled():
            task.exception()
        if self._tasks.get(key) is task:
            self._done_at[key] = self._monotonic()

    async def _fetch(self, name: str, sequence: int, url: str, nominal: float, edge_wait: float, progress: _Progress) -> _Fetched:
        progress.started = self._monotonic()
        data = await self._download(url, f"{name} sequence {sequence}", max(8.0, 4 * self._target), edge_wait, progress=progress)
        if data is None:
            return _Fetched(None, "not available (404)", is_404=True)
        segment = inspect_segment(name, sequence, data, nominal)
        if segment is None:
            return _Fetched(None, f"unreadable ({len(data)} bytes)")
        return _Fetched(segment)

    async def _wait(self, task: asyncio.Future[_Fetched], until: float) -> bool:
        """Waits for a fetch until `until` (monotonic), staying responsive to stop."""
        while not task.done():
            if self._stopping():
                raise _Closed()
            remaining = until - self._monotonic()
            if remaining <= 0:
                return False
            await self._wait_slice(task, min(0.2, remaining))
        if self._closed:
            # aclose() ran (from another task) while this waited: the fetches it was
            # tracking are gone.
            raise _Closed()
        return True

    async def _wait_slice(self, task: asyncio.Future[_Fetched], seconds: float) -> None:
        await asyncio.wait({task}, timeout=seconds)

    def _result(self, name: str, sequence: int, task: asyncio.Future[_Fetched]) -> _Fetched | None:
        """A finished fetch's outcome. For the clock track, a transient failure
        returns None (retried on the next call) until it has happened
        MAX_CONSECUTIVE_FETCH_FAILURES times in a row, then raises SourceError;
        a rejected URL raises SourceExpiredError. Other tracks never raise.
        A cancelled fetch counts as abandoned: its CancelledError is never
        re-raised, which would look like the caller itself being cancelled."""
        try:
            if task.cancelled():
                raise _Closed()
            fetched = task.result()
        except _Closed:
            if self._stopping() or name == self.clock_track:
                if task.cancelled():
                    self._drop((name, sequence))  # never reused: the sequence is fetched again on the next call
                raise
            return _Fetched(None, "abandoned")
        except SourceExpiredError as e:
            if name == self.clock_track:
                raise
            return _Fetched(None, str(e))
        except SourceError as e:
            if name != self.clock_track:
                return _Fetched(None, str(e))
            self._drop((name, sequence))  # retry this sequence
            self._fetch_failures += 1
            self.log(str(e))
            if self._fetch_failures >= self.MAX_CONSECUTIVE_FETCH_FAILURES:
                self._fetch_failures = 0
                raise
            return None
        except Exception as e:  # a bug in a parser must not kill the run
            return _Fetched(None, f"failed: {type(e).__name__}: {e}")
        if name == self.clock_track:
            self._fetch_failures = 0
        return fetched

    def _too_late(self, name: str, sequence: int) -> _Fetched:
        """Why a segment missed its grace period: still retrying (a 404 here usually
        means a stale URL), not started (its task hadn't run yet), no response yet
        (YouTube holding the request until it has the segment?), or a slow transfer."""
        progress = self._progress.get((name, sequence))
        if progress is None:
            return _Fetched(None, "too late")
        if progress.error:
            return _Fetched(None, f"too late (still retrying after {progress.error})", is_404=progress.error == "HTTP 404")
        if progress.started is None:
            return _Fetched(None, "too late (its download hadn't started)")
        elapsed = self._monotonic() - progress.started
        if not progress.responded:
            return _Fetched(None, f"too late (no response after {elapsed:.1f}s)")
        return _Fetched(None, f"too late ({progress.received / 1024:.0f} KiB received in {elapsed:.1f}s)")

    def _record(self, name: str, sequence: int, fetched: _Fetched, group: SegmentGroup) -> None:
        if fetched.segment is not None:
            group.segments[name] = fetched.segment
            self.consecutive_missing[name] = 0
            self._end_missing_streak(name, sequence)
            return
        if fetched.is_404 and name != self.clock_track and sequence not in self._tracks[name].urls:
            # A URL built from the template was refused: the template is probably stale.
            self._tracks[name].stale = True
        self.segments_missing += 1
        self.consecutive_missing[name] = self.consecutive_missing.get(name, 0) + 1
        if name not in self._missing_since:
            # Log the first of a streak; the streak's extent is logged when it ends.
            self._missing_since[name] = (sequence, fetched.reason)
            self.log(f"{name} sequence {sequence}: {fetched.reason}")

    def _end_missing_streak(self, name: str, sequence: int | None) -> None:
        start = self._missing_since.pop(name, None)
        if start is None:
            return
        first, reason = start
        last = (sequence - 1) if sequence is not None else (self._next_sequence or first + 1) - 1
        if last > first:
            self.log(f"{name}: sequences {first}-{last} were missing ({last - first + 1} segments; first reason: {reason})")
