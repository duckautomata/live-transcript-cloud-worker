"""Per-stream chunk pipeline: transcribe -> submit line -> hand off media.

Ordering rules (docs/01, docs/06):

- Line IDs are a strict gapless sequence assigned from local state.
- At most one in-flight POST /line per channel, submission is serialised.
- A 409 means the server disagrees; the only recovery is a full /sync.
- Media for line N is enqueued only after /line N returned.

Transcription requests, by contrast, are pure and can overlap. The pipeline
keeps up to ``transcription.concurrency`` provider calls in flight and
awaits them strictly in chunk order, so a provider hiccup on chunk N never
reorders lines, it just delays N+1's already-started request from being
consumed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time

from .audio import pcm_to_wav
from .config import Config, StreamerConfig
from .models import Chunk, Line, MediaType, Segment
from .server_client import LineResult
from .state import ChannelState
from .transcribe import TranscriptionService
from .uploader import MediaUploader

logger = logging.getLogger(__name__)

CHUNK_QUEUE_SIZE = 32
QUEUE_WARN_DEPTH = 10
LINE_LATENCY_WARN_SECONDS = 20.0

_END = None  # queue sentinel


class StreamPipeline:
    """Consumes one stream's chunks. Created per capture session."""

    def __init__(
        self,
        config: Config,
        streamer: StreamerConfig,
        state: ChannelState,
        server,
        transcriber: TranscriptionService,
        uploader: MediaUploader,
    ) -> None:
        self.config = config
        self.streamer = streamer
        self.state = state
        self.server = server
        self.transcriber = transcriber
        self.uploader = uploader
        self.queue: asyncio.Queue[Chunk | None] = asyncio.Queue(maxsize=CHUNK_QUEUE_SIZE)

    async def emit(self, chunk: Chunk) -> None:
        """Capture-side entry point; blocks when the queue is full so
        capture backpressure is explicit (docs/07: every queue is bounded)."""
        depth = self.queue.qsize()
        if depth >= QUEUE_WARN_DEPTH:
            logger.warning(
                "[%s] transcription backlog: %d chunks (~%.0fs of audio)",
                self.streamer.key,
                depth,
                depth * self.config.server.buffer_size_seconds,
            )
        await self.queue.put(chunk)

    async def finish(self) -> None:
        await self.queue.put(_END)

    async def run(self) -> None:
        """Drain the queue until the end sentinel, submitting lines in order."""
        window: asyncio.Queue[tuple[Chunk, asyncio.Task] | None] = asyncio.Queue(maxsize=max(1, self.config.transcription.concurrency))

        async def feeder() -> None:
            while True:
                chunk = await self.queue.get()
                if chunk is _END:
                    await window.put(None)
                    return
                task = asyncio.create_task(self._transcribe(chunk))
                try:
                    await window.put((chunk, task))
                except asyncio.CancelledError:
                    # Cancelled mid-handoff: the pair never reached the
                    # window, so nothing else can clean it up.
                    task.cancel()
                    chunk.path.unlink(missing_ok=True)
                    raise

        feeder_task = asyncio.create_task(feeder())
        try:
            while True:
                item = await window.get()
                if item is None:
                    break
                chunk, task = item
                try:
                    segments = await task
                    await self._submit(chunk, segments)
                except asyncio.CancelledError:
                    task.cancel()
                    chunk.path.unlink(missing_ok=True)
                    raise
                except Exception:
                    # An escaped exception here must never kill the consumer:
                    # capture blocks on emit() when the queue fills, so a dead
                    # consumer wedges the whole channel. Drop the chunk, keep
                    # consuming, and be loud about it.
                    logger.exception(
                        "[%s] pipeline error on chunk at %d; dropping chunk and continuing",
                        chunk.key,
                        int(chunk.audio_start_time),
                    )
                    task.cancel()
                    chunk.path.unlink(missing_ok=True)
        finally:
            feeder_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feeder_task
            # Never leak transcription tasks or chunk files on cancellation.
            while not window.empty():
                leftover = window.get_nowait()
                if leftover is not None:
                    leftover[1].cancel()
                    leftover[0].path.unlink(missing_ok=True)
            while not self.queue.empty():
                stranded = self.queue.get_nowait()
                if stranded is not None:
                    stranded.path.unlink(missing_ok=True)

    async def _transcribe(self, chunk: Chunk) -> list[Segment]:
        started = time.monotonic()
        relative = await self.transcriber.transcribe(pcm_to_wav(chunk.pcm))
        elapsed = time.monotonic() - started
        if relative is None:
            logger.error(
                "[%s] chunk at %d could not be transcribed by any provider; emitting empty line",
                chunk.key,
                int(chunk.audio_start_time),
            )
            relative = []
        logger.debug(
            "[%s] transcribed %.1fs chunk in %.2fs (%d segments)",
            chunk.key,
            chunk.duration,
            elapsed,
            len(relative),
        )
        segments = []
        for rel_start, text in relative:
            text = text.strip()
            if text:
                segments.append(Segment(timestamp=math.floor(chunk.audio_start_time + rel_start), text=text))
        return segments

    async def _submit(self, chunk: Chunk, segments: list[Segment]) -> None:
        """Append locally, POST, recover from 409 via full /sync, then hand
        the chunk file to the media queue (or delete it)."""
        line = Line(
            id=self.state.next_line_id,
            timestamp=math.floor(chunk.audio_start_time),
            segments=tuple(segments),
            vod_accurate=chunk.vod_accurate,
        )
        # Local state first: the /sync payload must already include this
        # line when a 409 comes back (docs/02).
        self.state.append_line(line)

        started = time.monotonic()
        result = await self.server.post_line(self.streamer.key, chunk.stream_id, line)
        delivered = result is LineResult.OK
        if result is LineResult.CONFLICT:
            logger.warning(
                "[%s] line %d conflicted; resyncing full state (%d lines)",
                chunk.key,
                line.id,
                len(self.state.transcript),
            )
            delivered = await self.server.sync(self.streamer.key, self.state.sync_payload())
            if not delivered:
                logger.error("[%s] resync failed; server remains behind until next success", chunk.key)
        elif result is LineResult.FAILED:
            logger.warning(
                "[%s] line %d not delivered; kept locally, a later 409->sync will repair",
                chunk.key,
                line.id,
            )

        latency = time.time() - (chunk.audio_start_time + chunk.duration)
        post_ms = (time.monotonic() - started) * 1000
        log = logger.warning if latency > LINE_LATENCY_WARN_SECONDS else logger.debug
        log(
            "[%s] line %d posted in %.0fms, %.1fs behind the audio edge",
            chunk.key,
            line.id,
            post_ms,
            latency,
        )

        wants_media = chunk.media_type is not MediaType.NONE and chunk.stream_id == self.state.stream_id
        if wants_media:
            try:
                queued = self.state.enqueue_media(chunk.stream_id, line.id, chunk.path)
            except OSError as exc:
                logger.warning("[%s] could not queue media for line %d: %s", chunk.key, line.id, exc)
            else:
                if delivered:
                    self.uploader.enqueue(self.streamer.key, chunk.stream_id, line.id, queued)
                else:
                    # Media for line N may only be uploaded after /line N
                    # landed (docs/06 invariant 4). The file stays parked in
                    # the durable queue; the boot-time rescan uploads it
                    # once a sync has repaired the server's transcript.
                    logger.info(
                        "[%s] parking media for undelivered line %d in the durable queue",
                        chunk.key,
                        line.id,
                    )
        else:
            chunk.path.unlink(missing_ok=True)
