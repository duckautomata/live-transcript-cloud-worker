"""YouTube Timeline segment sources: yt-dlp resolution (a scripted fake yt-dlp)
and HlsSegmentSource against a fake YouTube (httpx.MockTransport) on a fake clock."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from conftest import make_config
from timeline_fixtures import audio_duration, audio_segment, ts_filler_packet, video_segment

from live_transcript_cloud_worker import youtube_timeline_source as source_module
from live_transcript_cloud_worker.config import CaptureConfig
from live_transcript_cloud_worker.models import MediaType
from live_transcript_cloud_worker.youtube_timeline import TRACK_AUDIO, TRACK_MUXED, TRACK_VIDEO
from live_transcript_cloud_worker.youtube_timeline_source import (
    HLS_CLOCK_TO_VOD_SECONDS,
    HlsSegmentSource,
    ResolvedStream,
    ResolvedTrack,
    ResolveError,
    ResolveStopped,
    SourceContext,
    SourceError,
    SourceExpiredError,
    SourceResetError,
    StreamNotLiveError,
    StreamUnavailableError,
    UnsupportedSourceError,
    build_resolve_command,
    classify_resolve,
    create_segment_source,
    format_args,
    parse_media_playlist,
    parse_resolved,
    resolve_stream,
    sequence_url_template,
    url_changes,
    watch_url,
)

# ----------------------------------------------------------------------
# Small pure functions
# ----------------------------------------------------------------------


def test_watch_url_pins_the_video_id():
    assert watch_url("https://www.youtube.com/@chan/live", "9Mvg2efaRgo") == "https://www.youtube.com/watch?v=9Mvg2efaRgo"
    assert watch_url("https://www.youtube.com/@chan/live", "Unknown ID") == "https://www.youtube.com/@chan/live"


def test_format_args_require_hls_and_skip_video_when_not_needed():
    video = format_args(MediaType.VIDEO)
    assert video[1] == "bv[protocol^=m3u8]+ba[protocol^=m3u8]/b[protocol^=m3u8]"
    for media in (MediaType.AUDIO, MediaType.NONE):
        assert format_args(media)[1] == "ba[protocol^=m3u8]/b[protocol^=m3u8]"
    # Codec preferences stay a sort order, never a filter.
    assert video[2:] == ["-S", "vcodec:h264,proto:m3u8,acodec:aac"]


def test_build_resolve_command(tmp_path, monkeypatch):
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path="/bin/yt-dlp"))
    purposes = []

    def auth_args(config, url, purpose):
        purposes.append(purpose)
        return ["--cookies", "c.txt"]

    monkeypatch.setattr(source_module, "auth_args", auth_args)
    cmd = build_resolve_command(config, "https://www.youtube.com/watch?v=x", MediaType.VIDEO)
    assert cmd[:4] == ["/bin/yt-dlp", "-j", "--no-playlist", "--ignore-no-formats-error"]
    assert "--cookies" in cmd and cmd[-1] == "https://www.youtube.com/watch?v=x"
    assert purposes == ["download"]


def test_build_resolve_command_keeps_the_members_only_filter(tmp_path):
    config = make_config(tmp_path, capture=CaptureConfig(yt_dlp_path="/bin/yt-dlp"))
    cmd = build_resolve_command(config, "https://www.youtube.com/watch?v=x", MediaType.AUDIO)
    assert cmd[cmd.index("--match-filter") + 1] == "availability!=?subscriber_only"


PLAYLIST = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:1
#EXT-X-MEDIA-SEQUENCE:8092
#EXT-X-DISCONTINUITY-SEQUENCE:1
#EXT-X-PROGRAM-DATE-TIME:2026-10-14T01:15:00.000+00:00
#EXTINF:1.0,
https://rr3.googlevideo.com/videoplayback/id/x.1/itag/234/playlist/index.m3u8/sq/8092/goap/l/dur/1.000/file/seg.ts
#EXTINF:0.998,
https://rr3.googlevideo.com/videoplayback/id/x.1/itag/234/playlist/index.m3u8/sq/8093/goap/l/dur/1.000/file/seg.ts
#EXTINF:1.0,
seg-8094.ts
"""


def test_parse_media_playlist():
    pl = parse_media_playlist(PLAYLIST, "https://host/path/index.m3u8")
    assert pl is not None
    assert pl.media_sequence == 8092 and pl.target_duration == 1.0 and not pl.ended
    assert [s[0] for s in pl.segments] == [8092, 8093, 8094]
    assert pl.segments[1][1] == pytest.approx(0.998)
    assert pl.segments[2][2] == "https://host/path/seg-8094.ts"
    assert (pl.first_sequence, pl.last_sequence) == (8092, 8094)


def test_parse_media_playlist_endlist_and_rejects():
    pl = parse_media_playlist(PLAYLIST + "#EXT-X-ENDLIST\n", "https://host/")
    assert pl is not None and pl.ended
    assert parse_media_playlist("<html>", "https://host/") is None
    assert parse_media_playlist("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nlow.m3u8\n", "https://host/") is None
    empty = parse_media_playlist("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:5\n", "https://host/")
    assert empty is not None and empty.last_sequence is None


def test_sequence_url_template():
    url = "https://rr3.googlevideo.com/videoplayback/id/x.1/itag/234/playlist/index.m3u8/sq/8092/goap/l%3D1/dur/1.000/file/seg.ts"
    template = sequence_url_template(url, 8092)
    assert template is not None
    assert template.format(294) == url.replace("/sq/8092/", "/sq/294/")
    # The sequence in the URL must match the playlist's, or the template could fetch the wrong segment.
    assert sequence_url_template(url, 8093) is None
    assert sequence_url_template("https://host/seg-1.ts", 1) is None


def test_url_changes_names_the_changed_parts():
    base = "https://rr3.googlevideo.com/videoplayback/id/x.1/itag/234/sq/{}/goap/lmt%3D33/dur/2.000/file/seg.ts"
    assert url_changes(base, base.replace("lmt%3D33", "lmt%3D34")) == "goap: lmt%3D33 -> lmt%3D34"
    both = base.replace("lmt%3D33", "lmt%3D34").replace("2.000", "1.712")
    assert url_changes(base, both) == "goap: lmt%3D33 -> lmt%3D34, dur: 2.000 -> 1.712"
    assert url_changes(base, base + "/extra") == "a different path layout"
    many = "/".join(f"k{i}/v{i}" for i in range(5))
    assert url_changes(many, many.replace("/v", "/w")) == "k0: v0 -> w0, k1: v1 -> w1, k2: v2 -> w2 and 2 more"


def test_sequence_url_template_escapes_braces():
    template = sequence_url_template("https://h/v/sq/7/x/{weird}/seg.ts", 7)
    assert template is not None and template.format(9) == "https://h/v/sq/9/x/{weird}/seg.ts"


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------

V_URL = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1791954750/id/x.1/itag/312/playlist/index.m3u8"
A_URL = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1791954700/id/x.1/itag/234/playlist/index.m3u8"


def _metadata(**overrides):
    meta = {
        "id": "x",
        "live_status": "is_live",
        "requested_formats": [
            {"format_id": "312", "protocol": "m3u8_native", "url": V_URL, "vcodec": "avc1.64002A", "acodec": "none",
             "http_headers": {"User-Agent": "UA"}},
            {"format_id": "234", "protocol": "m3u8_native", "url": A_URL, "vcodec": "none", "acodec": None,
             "http_headers": {"User-Agent": "UA"}},
        ],
    }  # fmt: skip
    meta.update(overrides)
    return meta


def test_parse_resolved_split_hls():
    r = parse_resolved(_metadata())
    assert [t.track for t in r.tracks] == [TRACK_VIDEO, TRACK_AUDIO]
    assert r.track(TRACK_AUDIO).format_id == "234"  # type: ignore[union-attr]
    assert r.tracks[0].http_headers == {"User-Agent": "UA"}
    assert r.tracks[1].raw["format_id"] == "234"  # kept for future transports
    assert r.expires_at == 1791954700  # the earliest expiry wins


def test_parse_resolved_query_style_expiry():
    meta = _metadata()
    for f in meta["requested_formats"]:
        f["url"] = "https://host/videoplayback?expire=1791950000&itag=1"
    assert parse_resolved(meta).expires_at == 1791950000


def test_parse_resolved_single_muxed_format():
    meta = {"id": "x", "live_status": "is_live", "format_id": "95", "protocol": "m3u8_native", "url": V_URL,
            "vcodec": "avc1", "acodec": "mp4a.40.2"}  # fmt: skip
    assert [t.track for t in parse_resolved(meta).tracks] == [TRACK_MUXED]


def test_parse_resolved_audio_only():
    meta = {"id": "x", "live_status": "is_live", "format_id": "234", "protocol": "m3u8_native", "url": A_URL,
            "vcodec": "none", "acodec": None}  # fmt: skip
    assert [t.track for t in parse_resolved(meta).tracks] == [TRACK_AUDIO]


@pytest.mark.parametrize("status", ["was_live", "post_live", "not_live", "is_upcoming"])
def test_parse_resolved_not_live(status):
    with pytest.raises(StreamNotLiveError):
        parse_resolved(_metadata(live_status=status))


def test_parse_resolved_without_a_live_status_is_retried_not_ended():
    """With --ignore-no-formats-error a bot check comes back as JSON without a
    live status; that must not end the run as 'not live'."""
    with pytest.raises(ResolveError) as excinfo:
        parse_resolved({"id": "x", "live_status": None, "formats": []}, "WARNING: [youtube] x: Sign in to confirm you're not a bot")
    assert not isinstance(excinfo.value, (StreamNotLiveError, UnsupportedSourceError))


def test_parse_resolved_rate_limited_live_stream_is_retried():
    with pytest.raises(ResolveError) as excinfo:
        parse_resolved({"id": "x", "live_status": "is_live", "formats": []}, "WARNING: This content isn't available, try again later")
    assert not isinstance(excinfo.value, UnsupportedSourceError)


def test_parse_resolved_unavailable_warning_with_exit_zero():
    with pytest.raises(StreamUnavailableError):
        classify_resolve(
            0,
            json.dumps({"id": "x", "live_status": "is_live", "formats": []}),
            "WARNING: Join this channel to get access to members-only content",
        )


def test_parse_resolved_no_selection_tells_why():
    """With --ignore-no-formats-error yt-dlp still prints the JSON when no HLS
    format matched; the reason decides between waiting and retrying."""
    sabr = {"id": "x", "live_status": "is_live", "formats": [{"format_id": "140", "protocol": "sabr"}]}
    with pytest.raises(UnsupportedSourceError, match="SABR"):
        parse_resolved(sabr)
    with pytest.raises(ResolveError) as excinfo:
        parse_resolved({"id": "x", "live_status": "is_live", "formats": []}, "WARNING: [youtube] x: n challenge solving failed")
    assert not isinstance(excinfo.value, UnsupportedSourceError)
    with pytest.raises(UnsupportedSourceError, match="no HLS"):
        parse_resolved({"id": "x", "live_status": "is_live", "formats": [{"protocol": "https"}]})


def test_parse_resolved_rejects_non_hls_selection():
    meta = _metadata()
    meta["requested_formats"][1]["protocol"] = "sabr"
    with pytest.raises(UnsupportedSourceError, match="SABR"):
        parse_resolved(meta)
    meta["requested_formats"][1]["protocol"] = "https"
    with pytest.raises(UnsupportedSourceError, match="not HLS"):
        parse_resolved(meta)


def test_parse_resolved_requires_audio():
    meta = _metadata()
    meta["requested_formats"] = meta["requested_formats"][:1]
    with pytest.raises(UnsupportedSourceError, match="no audio"):
        parse_resolved(meta)


@pytest.mark.parametrize(
    "returncode, stdout, stderr, error",
    [
        (1, "", "ERROR: [youtube] x: This live event has ended.", StreamNotLiveError),
        (1, "", "ERROR: [youtube:tab] chan is not currently live", StreamNotLiveError),
        (1, "", "ERROR: [youtube] x: Join this channel to get access to members-only content", StreamUnavailableError),
        (1, "", "ERROR: unable to download webpage: timed out", ResolveError),
        (0, "", "", StreamUnavailableError),  # skipped by --match-filter
        (0, "not json", "", ResolveError),
        (0, "[1, 2]", "", ResolveError),
    ],
)
def test_classify_resolve_errors(returncode, stdout, stderr, error):
    with pytest.raises(error):
        classify_resolve(returncode, stdout, stderr)


def test_classify_resolve_success():
    assert len(classify_resolve(0, json.dumps(_metadata()), "").tracks) == 2


# -- resolve_stream with a scripted fake yt-dlp ---------------------------

needs_bash = pytest.mark.skipif(sys.platform == "win32", reason="the fake yt-dlp is a bash script")


def fake_ytdlp(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake-yt-dlp"
    script.write_text(f"#!/bin/bash\n{body}\n")
    script.chmod(0o755)
    return script


def _ytdlp_config(tmp_path: Path, ytdlp: Path | str):
    return make_config(tmp_path, capture=CaptureConfig(yt_dlp_path=str(ytdlp)))


class Stop:
    """A should_stop(): returns the scripted answers in order, then keeps the last one."""

    def __init__(self, values=None):
        self.values = list(values or [])
        self.value = False
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        if self.values:
            self.value = self.values.pop(0)
        return self.value


def _alive(pid: int) -> bool:
    """Whether a process still runs (a zombie waiting to be reaped doesn't)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


async def _wait_dead(pid: int, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return not _alive(pid)


async def _wait_for_file(path: Path, seconds: float = 5.0) -> None:
    end = time.monotonic() + seconds
    while not (path.exists() and path.read_text().strip()) and time.monotonic() < end:
        await asyncio.sleep(0.02)


@needs_bash
async def test_resolve_stream_success_and_logs_stderr(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(_metadata()))
    args = tmp_path / "args.txt"
    ytdlp = fake_ytdlp(
        tmp_path,
        f'printf "%s\\n" "$@" > "{args}"\ncat "{meta}"\necho "[youtube] x: Downloading webpage" >&2\nsleep 0.5',
    )
    logged: list[str] = []
    r = await resolve_stream(_ytdlp_config(tmp_path, ytdlp), "https://www.youtube.com/watch?v=x", MediaType.VIDEO, Stop(), logged.append)
    assert len(r.tracks) == 2
    assert logged == ["[youtube] x: Downloading webpage\n"]
    argv = args.read_text().splitlines()
    assert "--ignore-no-formats-error" in argv and argv[-1] == "https://www.youtube.com/watch?v=x"


@needs_bash
async def test_resolve_stream_not_live(tmp_path):
    ytdlp = fake_ytdlp(tmp_path, 'echo "ERROR: [youtube] x: This live event has ended." >&2\nexit 1')
    logged: list[str] = []
    with pytest.raises(StreamNotLiveError, match="has ended"):
        await resolve_stream(_ytdlp_config(tmp_path, ytdlp), "u", MediaType.AUDIO, Stop(), logged.append)
    assert logged and "has ended" in logged[0]


@needs_bash
async def test_resolve_stream_members_only_prints_no_json(tmp_path):
    """The --match-filter rejects members-only videos: exit 0, nothing on stdout."""
    ytdlp = fake_ytdlp(tmp_path, 'echo "[download] x does not pass filter (availability!=?subscriber_only), skipping .." >&2\nexit 0')
    with pytest.raises(StreamUnavailableError, match="does not pass filter"):
        await resolve_stream(_ytdlp_config(tmp_path, ytdlp), "u", MediaType.AUDIO, Stop())


@needs_bash
async def test_resolve_stream_stops_and_kills_the_process_group(tmp_path):
    """yt-dlp runs in its own session; on stop its whole group dies, children
    (deno) included, not just yt-dlp itself."""
    pids = tmp_path / "pids"
    ytdlp = fake_ytdlp(tmp_path, f'sleep 300 &\necho "$$ $!" > "{pids}"\nexec sleep 300')
    stop = Stop()
    task = asyncio.create_task(resolve_stream(_ytdlp_config(tmp_path, ytdlp), "u", MediaType.VIDEO, stop))
    await _wait_for_file(pids)
    leader, child = (int(p) for p in pids.read_text().split())
    assert _alive(leader) and _alive(child)
    started = time.monotonic()
    stop.value = True
    with pytest.raises(ResolveStopped):
        await asyncio.wait_for(task, timeout=10)
    assert time.monotonic() - started < 2  # polled at 0.2 s, killed at once
    assert await _wait_dead(leader) and await _wait_dead(child)


@needs_bash
async def test_resolve_stream_timeout_kills(tmp_path):
    pids = tmp_path / "pids"
    ytdlp = fake_ytdlp(tmp_path, f'sleep 300 &\necho "$$ $!" > "{pids}"\nexec sleep 300')
    with pytest.raises(ResolveError, match="timed out"):
        await asyncio.wait_for(resolve_stream(_ytdlp_config(tmp_path, ytdlp), "u", MediaType.VIDEO, Stop(), timeout=0.5), timeout=10)
    leader, child = (int(p) for p in pids.read_text().split())
    assert await _wait_dead(leader) and await _wait_dead(child)


@needs_bash
async def test_resolve_stream_cancelled_kills(tmp_path):
    pids = tmp_path / "pids"
    ytdlp = fake_ytdlp(tmp_path, f'sleep 300 &\necho "$$ $!" > "{pids}"\nexec sleep 300')
    task = asyncio.create_task(resolve_stream(_ytdlp_config(tmp_path, ytdlp), "u", MediaType.VIDEO, Stop()))
    await _wait_for_file(pids)
    leader, child = (int(p) for p in pids.read_text().split())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _wait_dead(leader) and await _wait_dead(child)


async def test_resolve_stream_not_started(tmp_path):
    with pytest.raises(ResolveError, match="could not be started"):
        await resolve_stream(_ytdlp_config(tmp_path, tmp_path / "no-such-yt-dlp"), "u", MediaType.VIDEO, Stop())


def _ctx(stop=None, max_lag=600.0):
    events: list[str] = []
    ctx = SourceContext(
        key="key",
        watch_url="u",
        media_type=MediaType.VIDEO,
        should_stop=stop or Stop(),
        log=events.append,
        max_lag_seconds=max_lag,
    )
    return ctx, events


async def test_create_segment_source_dispatch():
    ctx, _ = _ctx()
    hls = ResolvedStream("x", "is_live", (ResolvedTrack(TRACK_AUDIO, "m3u8_native", A_URL, "234"),), None)
    source = create_segment_source(hls, None, ctx)
    assert isinstance(source, HlsSegmentSource)
    assert source.vod_offset == HLS_CLOCK_TO_VOD_SECONDS and source.wrap_period
    await source.aclose()
    sabr = ResolvedStream("x", "is_live", (ResolvedTrack(TRACK_AUDIO, "sabr", "u", "140"),), None)
    with pytest.raises(UnsupportedSourceError):
        create_segment_source(sabr, None, ctx)


# ----------------------------------------------------------------------
# HlsSegmentSource with a fake YouTube
# ----------------------------------------------------------------------

SEG = audio_duration(43)
BASE = "https://rr.googlevideo.com/videoplayback/itag/{itag}/playlist/index.m3u8/sq/{sq}/goap/lmt%3D{lmt}/dur/1.000/file/seg.ts"


def _clock(sq):
    return 1000.0 + sq * SEG


class FakeStream(httpx.AsyncByteStream):
    """A response body delivered `piece` bytes at a time. `on_piece` runs before
    each piece (e.g. to let fake time pass); with `wait_between` each piece is a
    real wait on the event loop, as on a network."""

    def __init__(self, body: bytes, piece: int, on_piece=None, wait_between: bool = False):
        self._body = body
        self._piece = piece
        self._on_piece = on_piece
        self._wait_between = wait_between
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for i in range(0, len(self._body), self._piece):
            if self._wait_between:
                await asyncio.sleep(0)
            if self._on_piece:
                self._on_piece()
            self.reads += 1
            yield self._body[i : i + self._piece]

    async def aclose(self) -> None:
        self.closed = True


class FakeYouTube:
    """Serves media playlists (a 3-segment window ending at `head`) and segments
    by sequence number, with scriptable failures, as an httpx.MockTransport
    handler. `client()` is the source's client_factory."""

    def __init__(self, head=10, tracks=(TRACK_VIDEO, TRACK_AUDIO)):
        self.head = dict.fromkeys(tracks, head)
        self.ended = False
        self.window = 3
        # YouTube's segment URLs carry an `lmt` that can change mid-stream; a URL
        # with any other lmt is a 404. track -> [(first sequence, lmt), ...]
        self.lmts: dict[str, list[tuple[int, str]]] = {t: [(0, "33")] for t in tracks}
        self.status: dict[str, list[int]] = {}  # url substring -> statuses to return first
        self.headers_for: dict[int, dict] = {}
        self.raise_for: dict[str, list[Exception]] = {}
        self.missing: set[tuple[str, int]] = set()
        self.unreadable: set[str] = set()
        self.requests: list[str] = []
        self.on_chunk = None
        self.video_filler_packets = 0  # pads video segments to a realistic size
        self.piece: dict[str, int] = {}  # url substring -> bytes per piece
        self.playlist_on_chunk = None
        self.wait_between_pieces = False
        # Requests that never get a response (a download that never finishes).
        self.hold: Callable[[str], bool] = lambda url: False
        self.intercept: Callable[[str], httpx.Response | None] | None = None
        self.cancelled: list[str] = []  # held requests whose download was cancelled
        self.streams: list[tuple[str, FakeStream]] = []
        self.clients: list[httpx.AsyncClient] = []

    @staticmethod
    def itag(track):
        return {TRACK_VIDEO: "312", TRACK_AUDIO: "234", TRACK_MUXED: "95"}[track]

    def playlist_url(self, track):
        return f"https://manifest/{self.itag(track)}/index.m3u8"

    def lmt_for(self, track, sq):
        return [lmt for first, lmt in self.lmts[track] if sq >= first][-1]

    def change_lmt(self, from_sequence, lmt, tracks=None):
        for track in tracks or self.head:
            self.lmts[track].append((from_sequence, lmt))

    def client(self) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        self.clients.append(client)
        return client

    def _playlist(self, track):
        head = self.head[track]
        lines = ["#EXTM3U", "#EXT-X-TARGETDURATION:1", f"#EXT-X-MEDIA-SEQUENCE:{head - self.window + 1}"]
        for sq in range(head - self.window + 1, head + 1):
            lines += ["#EXTINF:1.0,", BASE.format(itag=self.itag(track), sq=sq, lmt=self.lmt_for(track, sq))]
        if self.ended:
            lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines) + "\n"

    def _ok(self, url, body, on_chunk=None):
        piece = next((n for key, n in self.piece.items() if key in url), None)
        if piece is None and on_chunk is None and not self.wait_between_pieces:
            return httpx.Response(200, content=body)
        stream = FakeStream(body, piece or 65536, on_chunk, self.wait_between_pieces)
        self.streams.append((url, stream))
        return httpx.Response(200, headers={"Content-Length": str(len(body))}, stream=stream)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if self.hold(url):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(url)
                raise
        if self.intercept is not None:
            response = self.intercept(url)
            if response is not None:
                return response
        for key, errors in self.raise_for.items():
            if key in url and errors:
                raise errors.pop(0)
        for key, statuses in self.status.items():
            if key in url and statuses:
                status = statuses.pop(0)
                return httpx.Response(status, headers=self.headers_for.get(status))
        for track in self.head:
            if url == self.playlist_url(track):
                return self._ok(url, self._playlist(track).encode(), self.playlist_on_chunk)
            if f"/itag/{self.itag(track)}/" in url:
                sq = int(url.split("/sq/")[1].split("/")[0])
                lmt = re.search(r"lmt%3D(\d+)", url)
                if (track, sq) in self.missing or sq > self.head[track] or (lmt and lmt.group(1) != self.lmt_for(track, sq)):
                    return httpx.Response(404)
                if track in self.unreadable:
                    return httpx.Response(200, content=b"<html>oops</html>")
                if track == TRACK_AUDIO:
                    data = audio_segment(_clock(sq))
                else:
                    data = video_segment(_clock(sq) + 0.01) + ts_filler_packet(0x100) * self.video_filler_packets
                return self._ok(url, data, self.on_chunk)
        return httpx.Response(404)


class FakeTime:
    def __init__(self, yt=None, advance_head_every=None):
        self.t = 0.0
        self.yt = yt
        self.every = advance_head_every
        self._next_advance = advance_head_every

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        while self.yt is not None and self._next_advance is not None and self.t >= self._next_advance:
            for track in self.yt.head:
                self.yt.head[track] += 1
            self._next_advance += self.every

    async def asleep(self, seconds):
        # Never suspends: a fetch task, once running, finishes in one go (MockTransport
        # doesn't suspend either), as deterministic as an inline executor.
        self.sleep(seconds)

    async def wait_slice(self, task, seconds):
        # Let ready fetch tasks run; time only passes for one that is still pending.
        for _ in range(5):
            if task.done():
                return
            await asyncio.sleep(0)
        if not task.done():
            self.sleep(seconds)


def _resolved(yt, tracks):
    return ResolvedStream("x", "is_live", tuple(ResolvedTrack(t, "m3u8_native", yt.playlist_url(t), yt.itag(t)) for t in tracks), None)


def _source(yt, resume=None, tracks=(TRACK_VIDEO, TRACK_AUDIO), stop=None, clock=None, max_lag=600.0):
    ctx, events = _ctx(stop, max_lag)
    src = HlsSegmentSource(_resolved(yt, tracks), resume, ctx, client_factory=yt.client)
    clock = clock or FakeTime()
    src._monotonic = clock.monotonic  # type: ignore[method-assign]
    src._sleep = clock.asleep  # type: ignore[method-assign]
    src._wait_slice = clock.wait_slice  # type: ignore[method-assign]
    return src, events, clock


async def _settle(rounds: int = 10) -> None:
    """Lets cancelled and ready tasks run."""
    for _ in range(rounds):
        await asyncio.sleep(0)


def _playlist_requests(yt, track):
    return sum(1 for u in yt.requests if u == yt.playlist_url(track))


async def test_source_starts_at_the_live_edge_and_pairs_tracks():
    yt = FakeYouTube(head=10)
    src, events, _ = _source(yt)
    group = await src.next_group(5)
    assert group is not None and group.sequence == 10
    assert set(group.segments) == {TRACK_VIDEO, TRACK_AUDIO}
    assert group.segments[TRACK_AUDIO].clock == pytest.approx(_clock(10), abs=1e-4)
    assert src.checkpoint == 11
    assert any("live edge, sequence 10" in e for e in events)
    await src.aclose()


async def test_source_starts_at_the_edge_every_track_lists():
    yt = FakeYouTube(head=10)
    yt.head[TRACK_VIDEO] = 9
    src, _, _ = _source(yt)
    group = await src.next_group(5)
    assert group is not None and group.sequence == 9
    await src.aclose()


async def test_source_polls_only_the_audio_playlist_after_opening():
    yt = FakeYouTube(head=10)
    clock = FakeTime(yt, advance_head_every=1.0)
    src, _, _ = _source(yt, clock=clock)
    for _ in range(5):
        assert await src.next_group(5) is not None
    assert _playlist_requests(yt, TRACK_VIDEO) == 1  # once, for the URL template
    assert _playlist_requests(yt, TRACK_AUDIO) > 1
    await src.aclose()


async def test_source_waits_for_the_next_sequence_then_returns_none_on_timeout():
    yt = FakeYouTube(head=10)
    src, _, clock = _source(yt, resume=11)
    assert await src.next_group(3) is None
    assert clock.t == pytest.approx(3, abs=1.01)
    yt.head = dict.fromkeys(yt.head, 11)
    group = await src.next_group(3)
    assert group is not None and group.sequence == 11
    await src.aclose()


async def test_source_follows_a_growing_playlist():
    yt = FakeYouTube(head=10)
    clock = FakeTime(yt, advance_head_every=1.0)
    src, _, _ = _source(yt, clock=clock)
    sequences = [(await src.next_group(5)).sequence for _ in range(5)]  # type: ignore[union-attr]
    assert sequences == [10, 11, 12, 13, 14]
    await src.aclose()


async def test_source_catches_up_by_number_after_falling_out_of_the_window():
    yt = FakeYouTube(head=20)
    src, events, _ = _source(yt, resume=12)
    group = await src.next_group(5)
    assert group is not None and group.sequence == 12 and set(group.segments) == {TRACK_VIDEO, TRACK_AUDIO}
    assert any("resuming at sequence 12" in e for e in events)
    await src.aclose()


async def test_source_jumps_to_the_live_edge_beyond_the_catch_up_window():
    yt = FakeYouTube(head=100)
    src, events, _ = _source(yt, resume=10, max_lag=30.0)
    group = await src.next_group(5)
    assert group is not None and group.sequence == 100
    assert src.segments_skipped == 90
    assert any("skipping 90 segments" in e for e in events)
    await src.aclose()


async def test_source_resume_before_a_stream_break_moves_to_the_oldest_available():
    """Segments from before a stream break return 404 by number: after a few in
    a row the source continues at the oldest segment the playlist lists."""
    yt = FakeYouTube(head=50)
    for sq in range(40, 48):
        yt.missing.add((TRACK_AUDIO, sq))
        yt.missing.add((TRACK_VIDEO, sq))
    src, events, _ = _source(yt, resume=40)
    groups = [await src.next_group(5) for _ in range(HlsSegmentSource.BACKFILL_404_LIMIT)]
    assert [g.sequence for g in groups[:-1] if g] == [40, 41]
    assert groups[-1] is None  # the jump itself
    group = await src.next_group(5)
    assert group is not None and group.sequence == 48 and TRACK_AUDIO in group.segments
    assert any("no longer available" in e for e in events)
    await src.aclose()


async def test_source_missing_video_segment_still_returns_audio_and_logs_the_streak_once():
    yt = FakeYouTube(head=10)
    clock = FakeTime(yt, advance_head_every=1.0)
    for sq in (10, 11, 12):
        yt.missing.add((TRACK_VIDEO, sq))
    src, events, _ = _source(yt, clock=clock)
    groups = [await src.next_group(5) for _ in range(4)]
    assert [set(g.segments) for g in groups] == [{TRACK_AUDIO}] * 3 + [{TRACK_AUDIO, TRACK_VIDEO}]  # type: ignore[union-attr]
    assert src.segments_missing == 3
    assert src.consecutive_missing == {TRACK_AUDIO: 0, TRACK_VIDEO: 0}
    video_events = [e for e in events if e.startswith("video")]
    assert len(video_events) == 2  # the first miss, then the streak summary
    assert "sequences 10-12 were missing" in video_events[1]
    await src.aclose()


async def test_source_close_logs_a_missing_streak_still_open():
    yt = FakeYouTube(head=11)
    yt.missing.update({(TRACK_VIDEO, 10), (TRACK_VIDEO, 11)})
    src, events, _ = _source(yt, resume=10)
    for _ in range(2):
        group = await src.next_group(5)
        assert group is not None and set(group.segments) == {TRACK_AUDIO}
    await src.aclose()
    assert events[-1] == "video: sequences 10-11 were missing (2 segments; first reason: not available (404))"


async def test_source_unreadable_audio_is_missing():
    yt = FakeYouTube(head=10)
    yt.unreadable.add(TRACK_AUDIO)
    src, events, _ = _source(yt)
    group = await src.next_group(5)
    assert group is not None and set(group.segments) == {TRACK_VIDEO}
    assert any("unreadable" in e for e in events)
    await src.aclose()


async def test_source_audio_403_means_expired_but_video_403_is_only_missing():
    yt = FakeYouTube(head=10)
    yt.status["/itag/234/"] = [403]
    src, _, _ = _source(yt)
    with pytest.raises(SourceExpiredError):
        await src.next_group(5)
    await src.aclose()

    yt = FakeYouTube(head=10)
    yt.status["/itag/312/"] = [403]
    src, events, _ = _source(yt)
    group = await src.next_group(5)
    assert group is not None and set(group.segments) == {TRACK_AUDIO}
    assert any("HTTP 403" in e for e in events)
    await src.aclose()


async def test_source_playlist_404_means_expired():
    yt = FakeYouTube(head=10)
    yt.status["manifest/234"] = [404, 404]
    src, _, _ = _source(yt)
    with pytest.raises(SourceExpiredError, match="gone"):
        await src.next_group(5)
    await src.aclose()


async def test_source_retries_transient_errors():
    yt = FakeYouTube(head=10)
    yt.raise_for["/itag/312/"] = [httpx.ConnectError("reset")]
    yt.status["/itag/234/"] = [503, 429]
    yt.headers_for[429] = {"Retry-After": "7"}
    src, _, clock = _source(yt)
    group = await src.next_group(15)
    assert group is not None and set(group.segments) == {TRACK_VIDEO, TRACK_AUDIO}
    assert clock.t >= 7.0  # honoured Retry-After (the default backoff would be 2 s)
    await src.aclose()


class BrokenStream(FakeStream):
    """A body that breaks after its first piece, as a dropped connection does."""

    def __init__(self, body: bytes, error: Exception):
        super().__init__(body, 10)
        self._error = error

    async def __aiter__(self):
        yield self._body[:10]
        raise self._error


def _break_mid_body(yt, needle, errors):
    """Segment responses for URLs containing `needle` start (200) and then break, once per error."""
    original = yt._ok

    def ok(url, body, on_chunk=None):
        if needle in url and "/sq/" in url and errors:
            return httpx.Response(200, headers={"Content-Length": str(len(body))}, stream=BrokenStream(body, errors.pop(0)))
        return original(url, body, on_chunk)

    yt._ok = ok


@pytest.mark.parametrize(
    "error",
    [httpx.RemoteProtocolError("peer closed connection"), httpx.ReadError("connection reset"), httpx.ReadTimeout("read timed out")],
    ids=["RemoteProtocolError", "ReadError", "ReadTimeout"],
)
async def test_source_retries_a_download_that_breaks_mid_body(error):
    """httpx raises these from the body stream, after the response started: the
    attempt must count as a transient failure (retried), not leave the segment out."""
    yt = FakeYouTube(head=10)
    errors = [error]
    _break_mid_body(yt, "/itag/234/", errors)
    src, _, _ = _source(yt)
    group = await src.next_group(15)
    assert errors == []  # the broken attempt happened
    assert group is not None and group.sequence == 10
    assert set(group.segments) == {TRACK_AUDIO, TRACK_VIDEO}
    await src.aclose()


async def test_source_downloads_that_keep_breaking_mid_body_raise():
    yt = FakeYouTube(head=10)
    _break_mid_body(yt, "/itag/234/", [httpx.RemoteProtocolError("peer closed")] * 1000)
    src, _, _ = _source(yt)
    for _ in range(HlsSegmentSource.MAX_CONSECUTIVE_FETCH_FAILURES - 1):
        assert await src.next_group(5) is None
    with pytest.raises(SourceError):
        await src.next_group(5)
    await src.aclose()


async def test_source_behind_live_counts_what_is_listed_and_how_old_the_listing_is():
    """The capture keeps its catch-up window open while the pipeline holds it, and
    estimates a missing start time, from this: listed sequences not delivered yet,
    plus the time since the playlist was read (the live edge doesn't wait)."""
    yt = FakeYouTube(head=10)
    src, _, clock = _source(yt, resume=5)
    assert src.behind_live is None  # nothing read yet
    group = await src.next_group(5)
    assert group is not None and group.sequence == 5
    listed = src.behind_live
    assert listed is not None and 5.0 <= listed < 6.0  # 6..10 at the 1 s target, read just now
    clock.sleep(30.0)  # the capture is busy elsewhere; the playlist isn't re-read
    assert src.behind_live == pytest.approx(listed + 30.0)
    await src.aclose()


async def test_source_edge_404_is_retried_until_the_segment_appears():
    yt = FakeYouTube(head=10)
    yt.status["/itag/234/playlist/index.m3u8/sq/10/"] = [404, 404]
    src, _, _ = _source(yt)
    group = await src.next_group(5)
    assert group is not None and TRACK_AUDIO in group.segments
    await src.aclose()


async def test_source_audio_failing_repeatedly_raises_without_consuming_the_sequence():
    yt = FakeYouTube(head=10)
    yt.raise_for["/itag/234/"] = [httpx.ReadTimeout("slow")] * 1000
    src, _, _ = _source(yt)
    for _ in range(HlsSegmentSource.MAX_CONSECUTIVE_FETCH_FAILURES - 1):
        assert await src.next_group(5) is None  # retried on the next call
        assert src.checkpoint == 10
    with pytest.raises(SourceError):
        await src.next_group(5)
    assert src.checkpoint == 10
    await src.aclose()


async def test_source_video_failing_repeatedly_never_raises():
    yt = FakeYouTube(head=10)
    yt.raise_for["/itag/312/"] = [httpx.ReadTimeout("slow")] * 1000
    clock = FakeTime(yt, advance_head_every=1.0)
    src, _, _ = _source(yt, clock=clock)
    for i in range(10):
        group = await src.next_group(30)
        assert group is not None and set(group.segments) == {TRACK_AUDIO}
        assert src.consecutive_missing[TRACK_VIDEO] == i + 1
    await src.aclose()


async def test_source_trickling_download_hits_its_deadline():
    yt = FakeYouTube(head=10)
    yt.video_filler_packets = 1500  # ~280 KB
    clock = FakeTime()
    yt.on_chunk = lambda: clock.sleep(5.0)  # every 64 KiB takes 5 s
    src, _, _ = _source(yt, tracks=(TRACK_VIDEO, TRACK_AUDIO), clock=clock)
    group = await src.next_group(60)
    # The 16 KiB audio segment fits in one piece; the large video segment can't finish in time.
    assert group is not None and set(group.segments) == {TRACK_AUDIO}
    await src.aclose()


async def test_source_late_video_is_left_out_after_the_grace_period():
    yt = FakeYouTube(head=10)
    yt.hold = lambda url: "/itag/312/" in url
    src, events, clock = _source(yt)
    group = await src.next_group(10)
    assert group is not None and set(group.segments) == {TRACK_AUDIO}
    assert clock.t <= 10
    await _settle()
    assert len(yt.cancelled) == 1 and "/sq/10/" in yt.cancelled[0]  # the download was abandoned
    assert any("too late" in e for e in events)
    await src.aclose()


async def test_source_late_video_within_grace_returns_none_first():
    yt = FakeYouTube(head=10)
    yt.hold = lambda url: "/itag/312/" in url
    src, _, _ = _source(yt)
    assert await src.next_group(0.5) is None  # grace (2 s) not over yet
    group = await src.next_group(5)
    assert group is not None and group.sequence == 10
    await src.aclose()


async def test_source_detects_endlist():
    yt = FakeYouTube(head=10)
    src, _, _ = _source(yt, resume=11)
    yt.ended = True
    assert await src.next_group(5) is None
    assert src.ended
    await src.aclose()


async def test_source_detects_a_sequence_reset():
    yt = FakeYouTube(head=11000)
    src, _, clock = _source(yt, resume=11001)
    yt.head = dict.fromkeys(yt.head, 30)  # numbering restarted: 11000 is gone too
    with pytest.raises(SourceResetError):
        await src.next_group(30)
    # It takes several polls over at least 10 s, not a single stale response.
    assert clock.t >= HlsSegmentSource.RESET_MIN_SECONDS
    await src.aclose()


async def test_source_stale_playlist_is_not_mistaken_for_a_reset():
    """An edge node serving an old playlist after a reconnect: the last delivered
    sequence is still served by number, so it's only stale."""
    yt = FakeYouTube(head=110)
    src, events, _ = _source(yt, resume=111)
    yt.head = dict.fromkeys(yt.head, 100)  # stale playlist...
    # ...but the segment it doesn't list yet still exists
    yt.intercept = lambda url: httpx.Response(200, content=audio_segment(_clock(110))) if "/sq/110/" in url else None
    assert await src.next_group(20) is None
    assert any("playlist is stale" in e for e in events)
    yt.head = dict.fromkeys(yt.head, 111)
    group = await src.next_group(20)
    assert group is not None and group.sequence == 111
    await src.aclose()


async def test_source_a_single_stale_poll_never_resets():
    yt = FakeYouTube(head=110)
    src, _, _ = _source(yt, resume=111)
    yt.head = dict.fromkeys(yt.head, 100)
    assert await src.next_group(1.5) is None
    yt.head = dict.fromkeys(yt.head, 111)
    assert await src.next_group(5) is not None
    await src.aclose()


async def test_source_stop_event_interrupts_waiting():
    yt = FakeYouTube(head=10)
    stop = Stop([False] * 4 + [True])
    src, _, clock = _source(yt, resume=11, stop=stop)
    assert await src.next_group(60) is None
    assert clock.t < 60
    await src.aclose()


async def test_source_audio_only_stream():
    yt = FakeYouTube(head=10, tracks=(TRACK_AUDIO,))
    src, _, _ = _source(yt, tracks=(TRACK_AUDIO,))
    group = await src.next_group(5)
    assert group is not None and set(group.segments) == {TRACK_AUDIO}
    assert src.clock_track == TRACK_AUDIO
    await src.aclose()


async def test_source_without_a_video_template_polls_the_video_playlist(monkeypatch):
    yt = FakeYouTube(head=10)
    clock = FakeTime(yt, advance_head_every=1.0)
    real = sequence_url_template
    monkeypatch.setattr(source_module, "sequence_url_template", lambda url, sq: None if "/itag/312/" in url else real(url, sq))
    src, _, _ = _source(yt, clock=clock)
    for _ in range(3):
        group = await src.next_group(5)
        assert group is not None and set(group.segments) == {TRACK_VIDEO, TRACK_AUDIO}
    assert _playlist_requests(yt, TRACK_VIDEO) > 1
    await src.aclose()


async def test_source_close_cancels_pending_fetches_and_closes_the_client():
    yt = FakeYouTube(head=10)
    yt.hold = lambda url: "/itag/312/" in url
    src, _, _ = _source(yt)
    assert await src.next_group(0.5) is None
    pending = list(src._running)
    assert pending  # the held video download
    await src.aclose()
    assert all(task.done() for task in pending) and not src._running
    assert len(yt.cancelled) == 1
    assert len(yt.clients) == 1 and yt.clients[0].is_closed
    assert await src.next_group(5) is None  # closed: returns at once
    await src.aclose()  # idempotent
    assert len(yt.clients) == 1  # closing never builds a new client


async def test_source_close_from_another_task_ends_a_waiting_next_group():
    yt = FakeYouTube(head=10)
    yt.hold = lambda url: "/itag/312/" in url
    src, _, _ = _source(yt)
    # Each wait lasts until the download finishes: next_group sees it end because aclose() cancelled it.
    src._wait_slice = lambda task, seconds: asyncio.wait({task})  # type: ignore[method-assign]
    waiting = asyncio.create_task(src.next_group(10))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    await asyncio.wait_for(src.aclose(), timeout=1)
    assert await asyncio.wait_for(waiting, timeout=1) is None
    assert len(yt.cancelled) == 1


async def test_source_close_before_any_request_builds_no_client():
    yt = FakeYouTube(head=10)
    src, _, _ = _source(yt)
    await src.aclose()
    await src.aclose()
    assert yt.clients == []


async def test_source_a_broken_video_playlist_never_stops_audio():
    for status in (503, 404, 403):
        yt = FakeYouTube(head=10)
        yt.status["manifest/312"] = [status] * 1000
        clock = FakeTime(yt, advance_head_every=1.0)
        src, events, _ = _source(yt, clock=clock)
        groups = [await src.next_group(30) for _ in range(5)]
        assert all(g is not None and set(g.segments) == {TRACK_AUDIO} for g in groups), status
        assert any("video playlist unavailable" in e for e in events)
        # The broken playlist is retried, but rarely.
        assert sum(1 for u in yt.requests if "manifest/312" in u) <= 3 * 3
        await src.aclose()


async def test_source_trickling_playlist_hits_its_deadline_and_stop_is_honoured():
    yt = FakeYouTube(head=10)
    clock = FakeTime()
    yt.piece["manifest/234"] = 100  # ~400 B playlist in 100 B pieces...
    yt.playlist_on_chunk = lambda: clock.sleep(2.0)  # ...2 s apart
    src, _, _ = _source(yt, clock=clock)
    with pytest.raises(SourceError):
        await src.next_group(60)
    # 3 attempts of a 4 s deadline plus backoff, not the full trickle each time.
    assert clock.t < 3 * (4 + 2) + 4
    await src.aclose()

    stop = Stop()
    yt = FakeYouTube(head=10)
    clock = FakeTime()
    yt.piece["manifest/234"] = 100

    def trickle():
        clock.sleep(2.0)
        stop.value = True

    yt.playlist_on_chunk = trickle
    src, _, _ = _source(yt, clock=clock, stop=stop)
    assert await src.next_group(60) is None
    assert clock.t <= 2.0
    await src.aclose()


async def test_source_a_body_that_completes_late_is_kept():
    yt = FakeYouTube(head=10)
    clock = FakeTime()
    # The (~1 KB) audio segment in 4 pieces, 2.5 s apart: the last one lands after the 8 s deadline.
    yt.piece["/itag/234/"] = -(-len(audio_segment(_clock(10))) // 4)
    yt.on_chunk = lambda: clock.sleep(2.5)
    src, _, _ = _source(yt, tracks=(TRACK_AUDIO,), clock=clock)
    group = await src.next_group(60)
    assert group is not None and TRACK_AUDIO in group.segments
    assert clock.t == pytest.approx(10.0)  # one attempt, kept although it finished past its deadline
    await src.aclose()


async def test_source_audio_is_not_starved_by_stalled_video_downloads():
    """Every video request hangs, audio is instant. Audio keeps flowing because
    every download is its own task, and each video download dropped as too late
    is cancelled instead of piling up."""
    yt = FakeYouTube(head=10)
    yt.hold = lambda url: "/itag/312/" in url
    src, _, clock = _source(yt)
    src.VIDEO_GRACE_TARGETS = 0.2  # type: ignore[misc]
    sequences = []
    while len(sequences) < 6 and clock.t < 60:
        yt.head = dict.fromkeys(yt.head, yt.head[TRACK_AUDIO] + 1)
        g = await src.next_group(2)
        if g is not None:
            assert TRACK_AUDIO in g.segments
            sequences.append(g.sequence)
    assert len(sequences) == 6
    await _settle()
    assert len(yt.cancelled) >= 5
    assert len(src._running) <= src.PREFETCH_SEQUENCES
    await src.aclose()


async def test_source_abandoned_video_download_stops_reading():
    """A video segment dropped as too late stops downloading instead of
    holding a connection and bandwidth."""
    yt = FakeYouTube(head=10)
    yt.video_filler_packets = 20_000  # ~3.7 MB
    yt.piece["/itag/312/"] = 1024
    yt.wait_between_pieces = True  # every piece is a wait on the network
    src, _, _ = _source(yt)
    group = None
    while group is None:
        group = await src.next_group(5)
    assert set(group.segments) == {TRACK_AUDIO}
    videos = [stream for url, stream in yt.streams if "/itag/312/" in url]
    assert videos
    await _settle()
    reads = [s.reads for s in videos]
    assert all(r > 0 for r in reads)
    await _settle(50)
    assert [s.reads for s in videos] == reads  # nothing reads any more
    assert all(s.closed for s in videos)
    await src.aclose()


async def test_source_video_grace_runs_from_when_the_audio_arrived():
    """Video that never arrives must not slow audio below real time: audio
    prefetched earlier has already used up its grace period."""
    yt = FakeYouTube(head=13)
    yt.hold = lambda url: "/itag/312/" in url
    src, _, clock = _source(yt, resume=10)
    groups = []
    while len(groups) < 4 and clock.t < 30:
        g = await src.next_group(5)
        if g is not None:
            groups.append(g)
    assert [g.sequence for g in groups] == [10, 11, 12, 13]
    assert all(set(g.segments) == {TRACK_AUDIO} for g in groups)
    # One grace period (2 s) for all four, not one each.
    assert clock.t < 3.0
    await src.aclose()


async def test_source_backfill_404s_are_counted_per_sequence_not_per_call():
    """Revisiting the same missing older segment (waiting on its video) must not
    count as several missing segments and trigger a jump past available ones."""
    yt = FakeYouTube(head=50)
    yt.missing.add((TRACK_AUDIO, 40))
    yt.hold = lambda url: "/itag/312/" in url and "/sq/40/" in url
    src, events, _ = _source(yt, resume=40)
    calls, group = 0, None
    while group is None:
        calls += 1
        group = await src.next_group(0.3)  # sequence 40 is revisited while its video is awaited
    assert calls > HlsSegmentSource.BACKFILL_404_LIMIT
    assert group.sequence == 40 and TRACK_AUDIO not in group.segments
    group = await src.next_group(5)
    assert group is not None and group.sequence == 41 and TRACK_AUDIO in group.segments
    assert not any("no longer available" in e for e in events)
    await src.aclose()


async def test_source_rate_limited_playlist_keeps_next_group_within_its_timeout():
    yt = FakeYouTube(head=10)
    src, _, clock = _source(yt, resume=10)
    assert await src.next_group(5) is not None  # opened, sequence 10 delivered
    yt.status["manifest/234"] = [429] * 100
    yt.headers_for[429] = {"Retry-After": "30"}
    started = clock.t
    assert await src.next_group(2) is None
    assert clock.t - started <= 2 + 1  # no 30 s Retry-After wait inside the call
    playlist_requests = _playlist_requests(yt, TRACK_AUDIO)
    assert await src.next_group(2) is None
    assert _playlist_requests(yt, TRACK_AUDIO) - playlist_requests <= 1  # not hammering either
    await src.aclose()


async def test_source_cancelled_fetch_never_looks_like_the_caller_being_cancelled():
    """A fetch task cancelled from outside is an abandoned download: a video one
    is left out of its group, a clock-track one is fetched again. Its
    CancelledError never escapes next_group."""
    yt = FakeYouTube(head=10)
    yt.hold = lambda url: "/itag/312/" in url
    src, events, _ = _source(yt)
    assert await src.next_group(0.5) is None  # video still within its grace period
    src._tasks[(TRACK_VIDEO, 10)].cancel()
    await _settle()
    group = await src.next_group(5)
    assert group is not None and group.sequence == 10 and set(group.segments) == {TRACK_AUDIO}
    assert "video sequence 10: abandoned" in events
    await src.aclose()

    yt = FakeYouTube(head=10, tracks=(TRACK_AUDIO,))
    holding = True
    yt.hold = lambda url: holding and "/itag/234/" in url
    src, _, _ = _source(yt, tracks=(TRACK_AUDIO,))
    assert await src.next_group(0.5) is None  # the audio segment hasn't arrived yet
    src._tasks[(TRACK_AUDIO, 10)].cancel()
    await _settle()
    holding = False
    assert await src.next_group(5) is None  # not a CancelledError
    group = await src.next_group(5)  # fetched again
    assert group is not None and group.sequence == 10 and TRACK_AUDIO in group.segments
    await src.aclose()


async def test_default_client_follows_redirects_with_the_resolved_headers():
    """googlevideo redirects; requests followed redirects by default, httpx doesn't."""
    yt = FakeYouTube(head=10)
    resolved = ResolvedStream(
        "x",
        "is_live",
        tuple(ResolvedTrack(t, "m3u8_native", yt.playlist_url(t), yt.itag(t), {"User-Agent": "UA"}) for t in (TRACK_VIDEO, TRACK_AUDIO)),
        None,
    )
    ctx, _ = _ctx()
    src = HlsSegmentSource(resolved, None, ctx)
    client = src._client_factory()
    try:
        assert client.follow_redirects is True
        assert client.headers["User-Agent"] == "UA"
        assert client.timeout == httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=5.0)
    finally:
        await client.aclose()
        await src.aclose()


async def test_default_client_follows_redirects_end_to_end(monkeypatch):
    """Every playlist and segment request is first redirected to another node;
    the source built with its default client still delivers the group."""
    yt = FakeYouTube(head=10)
    agents = set()
    redirects = []

    async def redirecting(request: httpx.Request) -> httpx.Response:
        agents.add(request.headers.get("User-Agent"))
        url = str(request.url)
        if request.url.host != "redirected.test":
            redirects.append(url)
            return httpx.Response(302, headers={"Location": "https://redirected.test/r/" + url.split("://", 1)[1]})
        return await yt.handler(httpx.Request("GET", "https://" + url.split("/r/", 1)[1]))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(source_module.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(redirecting), **kw))
    resolved = ResolvedStream(
        "x",
        "is_live",
        tuple(ResolvedTrack(t, "m3u8_native", yt.playlist_url(t), yt.itag(t), {"User-Agent": "UA"}) for t in (TRACK_VIDEO, TRACK_AUDIO)),
        None,
    )
    ctx, _ = _ctx()
    src = HlsSegmentSource(resolved, None, ctx)
    clock = FakeTime()
    src._monotonic = clock.monotonic  # type: ignore[method-assign]
    src._sleep = clock.asleep  # type: ignore[method-assign]
    src._wait_slice = clock.wait_slice  # type: ignore[method-assign]
    group = await src.next_group(5)
    assert group is not None and set(group.segments) == {TRACK_VIDEO, TRACK_AUDIO}
    assert len(redirects) == 4  # two playlists, two segments
    assert agents == {"UA"}
    await src.aclose()


# ----------------------------------------------------------------------
# URL templates going stale (seen in production: video lost after ~7 min)
# ----------------------------------------------------------------------


async def test_source_template_comes_from_the_newest_segment():
    """A DVR playlist spans minutes; its oldest segments can carry an lmt from
    before a mid-stream change. The template must follow the newest URL, or
    every video segment fetched by number is a 404."""
    yt = FakeYouTube(head=110)
    yt.window = 30
    yt.change_lmt(105, "34")
    clock = FakeTime(yt, advance_head_every=1.0)
    src, events, _ = _source(yt, clock=clock)
    groups = [await src.next_group(10) for _ in range(6)]
    assert [g.sequence for g in groups] == list(range(110, 116))  # type: ignore[union-attr]
    # 111+ aren't in the playlist read at open: built from the template, with the current lmt.
    assert all(set(g.segments) == {TRACK_VIDEO, TRACK_AUDIO} for g in groups)  # type: ignore[union-attr]
    await src.aclose()


async def test_source_follows_an_lmt_change_mid_stream():
    """The production failure: YouTube changes lmt while the worker runs. The
    audio playlist shows it; the video template is refreshed before it's used."""
    yt = FakeYouTube(head=10)
    yt.change_lmt(15, "34")
    clock = FakeTime(yt, advance_head_every=1.0)
    src, events, _ = _source(yt, clock=clock)
    groups = [await src.next_group(10) for _ in range(10)]
    assert [g.sequence for g in groups] == list(range(10, 20))  # type: ignore[union-attr]
    missing_video = [g.sequence for g in groups if TRACK_VIDEO not in g.segments]  # type: ignore[union-attr]
    assert missing_video == []
    assert any("segment URLs changed at sequence 15 (goap: lmt%3D33 -> lmt%3D34)" in e for e in events)
    await src.aclose()


async def test_source_video_404s_refresh_a_stale_template():
    """Safety net: only the video URLs change (the audio playlist can't show it).
    The 404s mark the template stale and the video playlist is re-read."""
    yt = FakeYouTube(head=10)
    yt.change_lmt(13, "34", tracks=[TRACK_VIDEO])
    clock = FakeTime(yt, advance_head_every=1.0)
    src, events, _ = _source(yt, clock=clock)
    groups = [await src.next_group(15) for _ in range(12)]
    missing_video = [g.sequence for g in groups if TRACK_VIDEO not in g.segments]  # type: ignore[union-attr]
    assert 0 < len(missing_video) <= 2  # the first refusal(s), then recovered
    assert TRACK_VIDEO in groups[-1].segments  # type: ignore[union-attr]
    assert any("video sequence 13" in e and "404" in e for e in events)
    await src.aclose()


def _stuck_video(src, clock, progress_fields):
    """Replaces the video downloads with ones stuck at the given progress forever."""
    real_fetch = src._fetch

    async def fetch(name, sequence, url, nominal, edge_wait, progress):
        if name != TRACK_VIDEO:
            return await real_fetch(name, sequence, url, nominal, edge_wait, progress)
        for field, value in progress_fields.items():
            setattr(progress, field, clock.monotonic() if field == "started" else value)
        await asyncio.Event().wait()

    src._fetch = fetch


@pytest.mark.parametrize(
    ("progress", "reason"),
    [
        ({}, "too late (its download hadn't started)"),
        ({"started": True}, "too late (no response after 2.0s)"),
        ({"started": True, "responded": True, "received": 150 * 1024}, "too late (150 KiB received in 2.0s)"),
    ],
)
async def test_too_late_says_how_far_the_download_got(progress, reason):
    """Tells YouTube holding a segment it doesn't have yet (no response) apart
    from a slow transfer or a download that never got going."""
    yt = FakeYouTube(head=10)
    src, events, clock = _source(yt)
    # The download is stuck at this point until the grace period (2 s here) is over.
    _stuck_video(src, clock, progress)
    group = await src.next_group(10)
    assert group is not None and set(group.segments) == {TRACK_AUDIO}
    assert any(e == f"video sequence 10: {reason}" for e in events), events
    await src.aclose()


async def test_too_late_names_what_the_download_keeps_getting():
    """'too late' used to hide a stream of 404s from a stale URL."""
    yt = FakeYouTube(head=10)
    src, events, clock = _source(yt)
    _stuck_video(src, clock, {"error": "HTTP 404"})  # the fetch's progress: it keeps getting 404s
    group = await src.next_group(10)
    assert group is not None and set(group.segments) == {TRACK_AUDIO}
    assert any("too late (still retrying after HTTP 404)" in e for e in events)
    # Sequence 10's URL came from the playlist itself: a 404 there isn't a template problem.
    assert not src._tracks[TRACK_VIDEO].stale
    yt.head[TRACK_AUDIO] = 11  # 11 isn't in the video playlist read at open: built from the template
    group = await src.next_group(10)
    assert group is not None and group.sequence == 11 and set(group.segments) == {TRACK_AUDIO}
    assert src._tracks[TRACK_VIDEO].stale  # so the video playlist is re-read before the next one
    await src.aclose()
