"""Media upload pool: concurrent, out-of-order-tolerant, channel-fair.

A small worker pool drains per-channel queues round-robin, so one channel's
backlog (or one slow R2 write) never starves another (docs/07 §4). Files are
streamed from disk into the multipart body.

File lifecycle: deleted on success and on server rejection (bounded-drop,
docs/04), but **kept and requeued with backoff when the server is
unreachable**, docs/06 requires that a server outage never destroys queued
media; the disk queue is bounded by the stream itself and is purged on
stream rotation.

Everything runs on one event loop; the synchronous queue bookkeeping needs
no locking, only a wakeup Event for idle workers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .server_client import MediaResult
from .state import ChannelState

logger = logging.getLogger(__name__)

RETRY_BACKOFF_BASE = 5.0
RETRY_BACKOFF_CAP = 120.0
IDLE_RECHECK_SECONDS = 1.0


@dataclass
class _Job:
    key: str
    stream_id: str
    line_id: int
    path: Path
    attempts: int = 0
    not_before: float = field(default=0.0)


class MediaUploader:
    def __init__(self, config: Config, server) -> None:
        self.config = config
        self.server = server
        self._queues: dict[str, deque[_Job]] = {}
        self._rr: deque[str] = deque()  # channels with pending work, round-robin
        self._work = asyncio.Event()
        self._workers: list[asyncio.Task] = []
        self._active = 0

    def start(self) -> None:
        count = max(1, self.config.server.media_upload_concurrency)
        self._workers = [asyncio.create_task(self._worker()) for _ in range(count)]

    def enqueue(self, key: str, stream_id: str, line_id: int, path: Path) -> None:
        self._push(_Job(key=key, stream_id=stream_id, line_id=line_id, path=path))

    def _push(self, job: _Job) -> None:
        queue = self._queues.setdefault(job.key, deque())
        queue.append(job)
        if job.key not in self._rr:
            self._rr.append(job.key)
        self._work.set()

    def rescan(self, states: dict[str, ChannelState]) -> int:
        """Re-enqueue media files left on disk by a previous run, ordered by
        line ID per channel; round-robin scheduling interleaves channels."""
        total = 0
        for key, state in states.items():
            for pending in state.pending_media():
                self.enqueue(key, pending.stream_id, pending.line_id, pending.path)
                total += 1
        if total:
            logger.info("re-enqueued %d media files from a previous run", total)
        return total

    def _next_job(self) -> _Job | None:
        now = asyncio.get_running_loop().time()
        for _ in range(len(self._rr)):
            key = self._rr.popleft()
            queue = self._queues.get(key)
            if not queue:
                continue
            if queue[0].not_before > now:
                self._rr.append(key)  # deferred; revisit on the next tick
                continue
            job = queue.popleft()
            if queue:
                self._rr.append(key)  # more work: back of the rotation
            return job
        return None

    async def _worker(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                self._work.clear()
                # Bounded wait: deferred (backing-off) jobs need a re-check
                # even when no new work arrives to set the event.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._work.wait(), timeout=IDLE_RECHECK_SECONDS)
                continue
            self._active += 1
            try:
                await self._upload(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A worker must never die: with all workers dead the disk
                # queue grows unboundedly and uploads silently stop.
                logger.exception(
                    "[%s] media upload %s/%d failed unexpectedly; file left on disk",
                    job.key,
                    job.stream_id,
                    job.line_id,
                )
            finally:
                self._active -= 1

    async def _upload(self, job: _Job) -> None:
        try:
            size = job.path.stat().st_size
        except OSError:
            return  # purged by a stream rotation; nothing to do
        started = time.monotonic()
        result = await self.server.upload_media(job.key, job.stream_id, job.line_id, job.path)
        elapsed_ms = (time.monotonic() - started) * 1000
        if result is MediaResult.OK:
            logger.debug(
                "[%s] media %s/%d uploaded in %.0fms (%.0f KB)",
                job.key,
                job.stream_id,
                job.line_id,
                elapsed_ms,
                size / 1024,
            )
            job.path.unlink(missing_ok=True)
        elif result is MediaResult.RETRY_LATER:
            job.attempts += 1
            delay = min(RETRY_BACKOFF_CAP, RETRY_BACKOFF_BASE * (2 ** min(job.attempts, 6)))
            job.not_before = asyncio.get_running_loop().time() + delay
            self._push(job)
            logger.info(
                "[%s] media %s/%d requeued (attempt %d, next try in %.0fs)",
                job.key,
                job.stream_id,
                job.line_id,
                job.attempts,
                delay,
            )
        else:  # DROP: permanent rejection, the file's journey ends here.
            job.path.unlink(missing_ok=True)

    def pending_count(self) -> int:
        return sum(len(q) for q in self._queues.values()) + self._active

    async def drain(self, timeout: float) -> bool:
        """Wait for the queues to empty. Returns True if fully drained;
        anything left stays on disk and is re-enqueued next boot."""
        deadline = asyncio.get_running_loop().time() + timeout
        while self.pending_count() > 0:
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.25)
        return True

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._workers = []
