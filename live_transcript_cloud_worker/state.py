"""Durable per-channel state.

Layout under ``tmp/{key}/``:

    meta.json          stream metadata (atomic)
    transcript.jsonl   append-only log, one JSON record per line (docs/07 §2)
    queue/             pending media uploads: {stream_id}__{line_id:06d}.ts
    segments/          live capture segment output (transient)
    ytdlp.log          yt-dlp stderr, tail-truncated
    stream_stats.log   raw probe output, tail-truncated

Each ``ChannelState`` is owned by exactly one task; nothing here is
thread-safe and nothing needs to be.

The in-memory transcript is the working copy; the full list is only
materialised into a payload when a /sync is actually needed. The append-only
log means a 6-hour stream costs 3600 small appends instead of 3600 rewrites
of a growing blob.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Line, MediaType

logger = logging.getLogger(__name__)

_QUEUE_FILE_RE = re.compile(r"^(?P<stream_id>[A-Za-z0-9_-]+)__(?P<line_id>\d+)\.ts$")


def atomic_write(path: Path, data: bytes) -> None:
    """write .tmp -> fsync -> rename; a crash never leaves a half file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class TailLog:
    """Append-mostly log file truncated to a bounded tail.

    Truncation rewrites the file **in place** (never rename-replaces it):
    capture subprocesses hold long-lived O_APPEND fds to this path, and a
    rename would strand their writes on an unlinked inode. O_APPEND writers
    simply continue at the new, shorter end.
    """

    def __init__(self, path: Path, max_bytes: int = 512 * 1024) -> None:
        self.path = path
        self.max_bytes = max_bytes

    def append(self, text: str) -> None:
        try:
            with open(self.path, "a", encoding="utf-8", errors="replace") as f:
                f.write(text if text.endswith("\n") else text + "\n")
            self.enforce_cap()
        except OSError as exc:  # diagnostics must never take down the pipeline
            logger.debug("tail log write failed for %s: %s", self.path, exc)

    def enforce_cap(self) -> None:
        try:
            if self.path.stat().st_size <= self.max_bytes:
                return
            data = self.path.read_bytes()[-self.max_bytes // 2 :]
            with open(self.path, "r+b") as f:
                f.write(data)
                f.truncate(len(data))
        except OSError as exc:
            logger.debug("tail log truncation failed for %s: %s", self.path, exc)


@dataclass(frozen=True)
class PendingMedia:
    stream_id: str
    line_id: int
    path: Path


class ChannelState:
    """All durable state for one channel, plus the in-memory transcript."""

    def __init__(self, key: str, root: Path) -> None:
        self.key = key
        self.root = root
        self.queue_dir = root / "queue"
        self.segments_dir = root / "segments"
        self.ytdlp_log = TailLog(root / "ytdlp.log")
        self.stream_stats_log = TailLog(root / "stream_stats.log")

        self.stream_id: str = ""
        self.stream_title: str = ""
        self.start_time: str = "0"  # string of unix seconds, per the wire format
        self.media_type: MediaType = MediaType.NONE
        self.is_live: bool = False
        self.transcript: list[Line] = []

        self.root.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(exist_ok=True)
        self.segments_dir.mkdir(exist_ok=True)
        self._load()

    # ---------------------------------------------------------------- meta

    @property
    def _meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def _transcript_path(self) -> Path:
        return self.root / "transcript.jsonl"

    @property
    def next_line_id(self) -> int:
        return len(self.transcript)

    def _load(self) -> None:
        try:
            meta = json.loads(self._meta_path.read_text())
        except FileNotFoundError:
            meta = {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[%s] meta.json unreadable (%s); starting fresh", self.key, exc)
            meta = {}
        self.stream_id = str(meta.get("stream_id", ""))
        self.stream_title = str(meta.get("stream_title", ""))
        self.start_time = str(meta.get("start_time", "0"))
        try:
            self.media_type = MediaType(meta.get("media_type", "none"))
        except ValueError:
            self.media_type = MediaType.NONE
        self.is_live = bool(meta.get("is_live", False))

        self.transcript = self._load_transcript()
        if self.stream_id:
            logger.info(
                "[%s] restored state: stream=%s lines=%d live=%s",
                self.key,
                self.stream_id,
                len(self.transcript),
                self.is_live,
            )

    def _load_transcript(self) -> list[Line]:
        """Read the valid prefix of the append-only log.

        Only the final record can be torn by a crash; any parse failure or
        ID discontinuity ends the readable prefix. The file is truncated back
        to that prefix so future appends stay consistent.
        """
        lines: list[Line] = []
        try:
            raw = self._transcript_path.read_bytes()
        except FileNotFoundError:
            return lines
        except OSError as exc:
            logger.warning("[%s] transcript log unreadable (%s); starting empty", self.key, exc)
            return lines

        valid_bytes = 0
        for record in raw.split(b"\n"):
            if not record.strip():
                valid_bytes += len(record) + 1
                continue
            try:
                line = Line.from_wire(json.loads(record))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                break
            if line.id != len(lines):
                logger.warning(
                    "[%s] transcript log discontinuity at id=%d (expected %d); truncating",
                    self.key,
                    line.id,
                    len(lines),
                )
                break
            lines.append(line)
            valid_bytes += len(record) + 1

        valid_bytes = min(valid_bytes, len(raw))
        if valid_bytes < len(raw):
            with open(self._transcript_path, "r+b") as f:
                f.truncate(valid_bytes)
        return lines

    def save_meta(self) -> None:
        meta = {
            "stream_id": self.stream_id,
            "stream_title": self.stream_title,
            "start_time": self.start_time,
            "media_type": self.media_type.value,
            "is_live": self.is_live,
        }
        atomic_write(self._meta_path, json.dumps(meta).encode())

    # ------------------------------------------------------------- streams

    def start_new_stream(self, stream_id: str, title: str, start_time: int, media_type: MediaType) -> None:
        """Reset everything for a different stream ID (docs/02 activate rules)."""
        self.stream_id = stream_id
        self.stream_title = title
        self.start_time = str(start_time)
        self.media_type = media_type
        self.is_live = True
        self.transcript = []
        with open(self._transcript_path, "wb"):
            pass  # truncate
        self.purge_media_queue()
        self.save_meta()

    def refresh_stream(self, title: str, start_time: int) -> None:
        """Same-ID reactivation: titles change mid-stream; keep the transcript."""
        self.stream_title = title
        self.start_time = str(start_time)
        self.is_live = True
        self.save_meta()

    def set_live(self, live: bool) -> None:
        if self.is_live != live:
            self.is_live = live
            self.save_meta()

    # ------------------------------------------------------------- lines

    def append_line(self, line: Line) -> None:
        if line.id != self.next_line_id:
            raise ValueError(f"line id {line.id} breaks the gapless sequence (expected {self.next_line_id})")
        record = json.dumps(line.to_wire(), separators=(",", ":")) + "\n"
        with open(self._transcript_path, "ab") as f:
            f.write(record.encode())
            f.flush()
            os.fsync(f.fileno())
        self.transcript.append(line)

    def sync_payload(self) -> dict[str, Any]:
        """The complete WorkerData body for POST /{channel}/sync."""
        return {
            "streamId": self.stream_id,
            "streamTitle": self.stream_title,
            "startTime": self.start_time,
            "isLive": self.is_live,
            "mediaType": self.media_type.value,
            "transcript": [line.to_wire() for line in self.transcript],
        }

    # ------------------------------------------------------------- media

    def enqueue_media(self, stream_id: str, line_id: int, src: Path) -> Path:
        """Move a finished chunk file into the durable upload queue.

        A rename on the same filesystem is atomic: a crash leaves either the
        source or the queued file, never a half file.
        """
        dest = self.queue_dir / f"{stream_id}__{line_id:06d}.ts"
        os.replace(src, dest)
        return dest

    def pending_media(self) -> list[PendingMedia]:
        """Queued uploads, ordered by line ID. Ignores foreign/.tmp files."""
        found: list[PendingMedia] = []
        try:
            entries = list(self.queue_dir.iterdir())
        except OSError:
            return found
        for path in entries:
            m = _QUEUE_FILE_RE.match(path.name)
            if not m:
                continue
            found.append(
                PendingMedia(
                    stream_id=m.group("stream_id"),
                    line_id=int(m.group("line_id")),
                    path=path,
                )
            )
        found.sort(key=lambda p: p.line_id)
        return found

    def purge_media_queue(self) -> None:
        """Drop queued media; it belongs to a stream the server rotated past."""
        for path in self.queue_dir.iterdir():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("[%s] could not purge %s: %s", self.key, path, exc)

    def clear_segments_dir(self) -> None:
        """Remove leftover transient segments from previous capture sessions.

        Only called at worker boot, when no capture session (and no pipeline
        holding chunk paths) can be alive.
        """
        for path in self.segments_dir.iterdir():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as exc:
                logger.warning("[%s] could not remove segment %s: %s", self.key, path, exc)
