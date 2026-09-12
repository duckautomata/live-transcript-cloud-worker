"""Shared data types.

The wire formats here mirror docs/01-server-api.md exactly:
- ``timestamp`` fields are integer unix seconds.
- ``startTime`` is a *string* of unix seconds in /sync and /activate.
- Segments are opaque to the server but the client expects
  ``[{"timestamp": <int>, "text": <str>}]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

STREAM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def valid_stream_id(stream_id: str) -> bool:
    return bool(STREAM_ID_RE.match(stream_id))


class MediaType(StrEnum):
    NONE = "none"
    AUDIO = "audio"
    VIDEO = "video"


class Platform(StrEnum):
    YOUTUBE = "youtube"
    TWITCH = "twitch"
    OTHER = "other"


@dataclass(frozen=True)
class Segment:
    """One transcribed span. ``timestamp`` is absolute unix seconds (int)."""

    timestamp: int
    text: str

    def to_wire(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "text": self.text}


@dataclass(frozen=True)
class Line:
    """One transcript line, exactly the /line body."""

    id: int
    timestamp: int
    segments: tuple[Segment, ...]
    vod_accurate: bool

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "segments": [s.to_wire() for s in self.segments],
            "mediaAvailable": False,
            "vodAccurate": self.vod_accurate,
        }

    @staticmethod
    def from_wire(data: dict[str, Any]) -> Line:
        return Line(
            id=int(data["id"]),
            timestamp=int(data["timestamp"]),
            segments=tuple(Segment(timestamp=int(s["timestamp"]), text=str(s["text"])) for s in data.get("segments", [])),
            vod_accurate=bool(data.get("vodAccurate", False)),
        )


# live_status values that mean "resolved, but definitively not a live stream
# right now" -- an ended broadcast or a plain video. Kept in sync with the
# StreamInfo.live_status comment.
_TERMINAL_OFFLINE_STATUSES = frozenset({"was_live", "post_live", "not_live"})


@dataclass(frozen=True)
class StreamInfo:
    """Result of probing a URL with yt-dlp."""

    url: str
    stream_id: str = ""
    title: str = ""
    is_live: bool = False
    live_status: str = "unknown"  # is_live | is_upcoming | was_live | post_live | not_live | unknown
    scheduled_start: int | None = None  # unix seconds, when live_status == is_upcoming
    start_time: int | None = None  # actual stream start (release_timestamp/timestamp)

    @property
    def is_upcoming(self) -> bool:
        return self.live_status == "is_upcoming"

    @property
    def is_terminal_offline(self) -> bool:
        """A definitive not-live verdict from a *successful* probe: the stream
        has ended (``was_live``/``post_live``) or the video was never a
        broadcast (``not_live``). As conclusive as yt-dlp's UserNotLive
        failure, so the watcher treats it as confirmed offline -- it backs off
        and, in incoming mode, advances the offline-delete threshold. An empty
        or ``unknown`` status is deliberately excluded: yt-dlp emits it while a
        stream is briefly between states, and acting on it could drop a URL
        that is about to go (or come back) live.
        """
        return self.live_status in _TERMINAL_OFFLINE_STATUSES


@dataclass(frozen=True)
class Chunk:
    """One captured media span destined to become exactly one line.

    ``path`` points at the MPEG-TS segment on disk; media bytes travel as
    paths, never as buffers (docs/07 design rule). ``pcm`` is the decoded
    16 kHz mono audio (~32 KB/s) produced by the same single decode pass
    that measured ``duration``; the transcription stage wraps it in a WAV
    header instead of decoding the segment a second time.
    """

    key: str
    stream_id: str
    path: Path
    audio_start_time: float  # unix seconds at which this chunk's audio begins
    duration: float  # measured from decoded samples, the master clock
    vod_accurate: bool
    media_type: MediaType
    pcm: bytes = field(repr=False, default=b"")


class ProbeResult(Enum):
    LIVE = "live"
    UPCOMING = "upcoming"
    OFFLINE = "offline"
    ERROR = "error"
