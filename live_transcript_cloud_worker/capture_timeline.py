"""YouTube Timeline capture: VOD-accurate YouTube lines from the live edge.

YouTube stamps every live segment with its media clock (TS PTS, or the ID3
timestamp of the raw ADTS audio rendition), and that clock is the VOD
timeline. This capture fetches the segments itself, pairs video and audio by
sequence number, and stamps each chunk with
``start_time + clock of its first audio sample + the source's VOD offset``,
so every line is VOD-accurate without downloading the stream from the start
(no --live-from-start, works with DVR disabled). Losing segments loses only
those lines: later ones still carry their own clock.

Video and audio are muxed per chunk with ffmpeg -copyts, so both keep the
clock and stay in sync (see youtube_timeline.build_mux_command). Each muxed
chunk is a file in the run's work dir, decoded once for its PCM and measured
duration (as live-edge capture does with its segments) and handed to the
pipeline by path.

Transport lives in youtube_timeline_source: HLS today, SABR to be added as
another SegmentSource.

Health: the connection (yt-dlp resolution + HTTP client) is rebuilt when the
URLs are rejected, the playlist stops advancing, segments keep arriving with
nothing usable (stale_threshold.no_audio_seconds), or the video track stays
missing; URLs are refreshed before they expire without dropping the current
connection. yt-dlp runs are paced and back off while reconnects bring
nothing. The run only ends when yt-dlp says the stream is no longer live (or
can't be watched), when the playlist ends, or on stop, so the watcher doesn't
flip the stream offline and back. A run resumes where the previous run of the
same stream stopped (tmp/{key}/youtube_timeline_state.json) when that is
within stale_threshold.lfs_gap_seconds of the live edge.
Diagnostics: tmp/{key}/youtube_timeline.log.

Start time: the server places a line at ``timestamp - startTime``, and the
watcher activates a stream with its probed start time, or with the activation
time when the probe had none. With a probed start time the clock lands lines
exactly where the VOD has them. Without one the clock's zero is estimated from
the first chunk's arrival, so lines still follow the stream clock smoothly,
but the server measures them from the activation time, which is not the
clock's zero: those lines are never marked VOD-accurate.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from . import audio
from .capture import ShouldStop
from .config import Config, StreamerConfig
from .models import Chunk, StreamInfo
from .state import ChannelState, TailLog, atomic_write
from .youtube_timeline import ChunkAssembler, ClockUnwrapper, MuxError, TimelineChunk, mux_chunk
from .youtube_timeline_source import (
    ResolvedStream,
    ResolveError,
    ResolveStopped,
    SegmentSource,
    SourceContext,
    SourceError,
    SourceExpiredError,
    SourceResetError,
    StreamNotLiveError,
    StreamUnavailableError,
    UnsupportedSourceError,
    create_segment_source,
    resolve_stream,
    watch_url,
)

logger = logging.getLogger(__name__)

Emit = Callable[[Chunk], Awaitable[None]]


class _EndRun(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Outcome:
    kind: str  # "end", "reconnect", "swap" (fresh URLs resolved, keep going) or "reset" (start over at the live edge)
    reason: str
    resolved: ResolvedStream | None = None


@dataclass
class TimelineStats:
    chunks_produced: int = 0
    chunks_failed: int = 0
    chunks_not_vod_accurate: int = 0
    gaps: int = 0
    reconnects: int = 0
    segments_missing: int = 0
    segments_skipped: int = 0


async def _no_run(chunk: Chunk) -> None:
    raise RuntimeError("TimelineCapture.run() is not active")


class TimelineCapture:
    """One per channel (the watcher keeps it), so yt-dlp pacing survives across runs of the same stream."""

    _NAME = "YouTube timeline"
    _LOG_FILENAME = "youtube_timeline.log"
    _STATE_FILENAME = "youtube_timeline_state.json"
    _MUX_SUBDIR = "mux"

    # How long one next_group() call may wait; keeps the checks below responsive.
    _POLL_TIMEOUT_SECONDS = 2.0
    # Refresh the URLs this long before they expire, keeping the current connection until that succeeds...
    _REFRESH_BEFORE_EXPIRY_SECONDS = 20 * 60
    _PROACTIVE_RETRY_SECONDS = 60.0
    # ...and reconnect regardless this close to expiry.
    _HARD_REFRESH_BEFORE_EXPIRY_SECONDS = 60.0
    # An expiry missing or this close after resolving isn't trusted: rejected URLs (403) take over.
    _MIN_EXPIRY_HEADROOM_SECONDS = 15 * 60
    # No new segment on a connection for this long: rebuild it (frozen playlist, ended stream).
    _IDLE_RECONNECT_SECONDS = 30.0
    # yt-dlp never runs more often than this...
    _MIN_RESOLVE_INTERVAL_SECONDS = 15.0
    # ...and backs off like this while it fails or reconnects bring nothing usable, but never
    # past half of lfs_gap_seconds: a stream that comes back from an outage right after a failed
    # attempt would otherwise be further behind than that by the next one, and get skipped.
    _BACKOFF_SECONDS = (15.0, 30.0, 60.0, 120.0, 300.0)
    # yt-dlp offered nothing readable (e.g. SABR-only): ask again this rarely.
    _UNSUPPORTED_RETRY_SECONDS = 600.0
    # A secondary track (video) missing this many sequences in a row triggers a reconnect, at most this often.
    _TRACK_RECONNECT_AFTER = 30
    _TRACK_RECONNECT_INTERVAL_SECONDS = 600.0
    _MAX_SOURCE_ERRORS = 3
    # Where start time + clock may put a chunk relative to now and still count as anchored to the
    # stream's start: up to this much later (start time recorded after the clock started, drift
    # over days)...
    _CLOCK_AHEAD_TOLERANCE_SECONDS = 900.0
    # ...or this much earlier on top of the catch-up window (delivery delay).
    _CLOCK_BEHIND_TOLERANCE_SECONDS = 300.0
    # A chunk starting within this of where the last VOD-accurate one ended continues its timeline.
    _CONTINUITY_TOLERANCE_SECONDS = 2.0
    # Used only when the stream's start time is unknown: delay between a frame and its arrival.
    _ASSUMED_DELIVERY_SECONDS = 2.0
    # Repeated warnings of one kind are logged at most this often.
    _WARN_INTERVAL_SECONDS = 300.0
    # Waiting this long on the pipeline is a backlog worth logging.
    _BACKLOG_LOG_WAIT_SECONDS = 1.0

    # Time sources (overridable in tests).
    _monotonic = staticmethod(time.monotonic)
    _wallclock = staticmethod(time.time)

    def __init__(self, config: Config, streamer: StreamerConfig, state: ChannelState) -> None:
        self.config = config
        self.streamer = streamer
        self.state = state
        # Bounded on every append (TailLog), so a long run never outgrows it.
        self._run_log = TailLog(state.root / self._LOG_FILENAME)
        self._state_path = state.root / self._STATE_FILENAME
        # yt-dlp pacing survives across runs of the same stream (the watcher may restart a run).
        self._pacing_stream_id: str | None = None
        self._next_resolve_at = 0.0
        self._backoff_step = 0
        self._should_stop: ShouldStop = lambda: False
        self._emit_chunk: Emit = _no_run
        self._reset_run()

    def _reset_run(self) -> None:
        self._stats = TimelineStats()
        self._work_dir = self.state.segments_dir / f"t{int(time.time())}_{os.getpid()}"
        self._chunk_number = 0
        self._start_time: float | None = None
        self._start_time_estimated = False
        self._vod_offset = 0.0
        self._warned_at: dict[str, float] = {}
        self._last_track_reconnect = -1e9
        self._unsupported_reason: str | None = None
        # Where the last VOD-accurate chunk ended: (clock, wall time it was queued, whether it
        # came from the state file). Chunks continuing that timeline stay VOD-accurate.
        self._anchor: tuple[float, float, bool] | None = None
        # Seconds spent waiting in emit() since the source was last within lfs_gap_seconds of
        # live (see _track_backlog); whether that backlog was logged, and the run's skipped
        # segments when it was.
        self._emit_wait = 0.0
        self._backlog_logged = False
        self._skipped_at_backlog = 0
        # Seconds of stream already published past the last group read (source.behind_live).
        self._behind_live = 0.0
        # The run's SourceContext, shared by all its sources: its max_lag_seconds is the window.
        self._ctx: SourceContext | None = None

    @property
    def _lfs_gap_seconds(self) -> float:
        return self.config.server.stale_threshold.lfs_gap_seconds

    @property
    def _catch_up_window(self) -> float:
        """How far behind live the source may fall before it skips to the live edge:
        lfs_gap_seconds, widened by the time spent waiting on a backed-up pipeline
        (see _track_backlog). Doubled: the source counts its lag in playlist target
        durations, which round segment lengths up, and the wait leaves out fetching
        and muxing what waited."""
        return self._lfs_gap_seconds + 2 * self._emit_wait

    def _update_catch_up_window(self) -> None:
        if self._ctx is not None:
            self._ctx.max_lag_seconds = self._catch_up_window

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self, info: StreamInfo, emit: Emit, should_stop: ShouldStop) -> TimelineStats:
        """Captures one live stream until it ends or should_stop() turns true.
        Every chunk goes out through ``emit`` (which may block for
        backpressure). Never raises, except CancelledError: a cancelled run
        only cleans up (no final chunk) and re-raises."""
        key = self.streamer.key
        logger.info("[%s] timeline: starting (stream %s)", key, info.stream_id)
        self._reset_run()
        self._should_stop = should_stop
        self._emit_chunk = emit
        if self._pacing_stream_id != info.stream_id:
            self._pacing_stream_id = info.stream_id
            self._next_resolve_at = 0.0
            self._backoff_step = 0

        self._run_log.append(f"\n--- {self._NAME} started at {time.strftime('%Y-%m-%d %H:%M:%S')} (stream: {info.stream_id}) ---")
        self._start_time = self._parse_start_time(info.start_time)
        if self._start_time is None:
            self._event(
                "the stream's start time is unknown, so it will be estimated from the first chunk. The server measures lines "
                "from when it activated the stream, not from the clock's zero, so they are not marked VOD-accurate",
                logging.WARNING,
            )

        ctx = SourceContext(
            key=key,
            watch_url=watch_url(info.url, info.stream_id),
            media_type=self.streamer.media_type,
            should_stop=should_stop,
            log=self._event,
            max_lag_seconds=float(self._lfs_gap_seconds),
        )
        self._ctx = ctx
        resume: int | None = None
        assembler: ChunkAssembler | None = None
        source: SegmentSource | None = None
        pending: ResolvedStream | None = None
        end_reason = "stopped"
        cancelled = False
        try:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            resume = self._load_state(info.stream_id)
            while not self._should_stop():
                resolved = pending if pending is not None else await self._resolve(ctx)
                pending = None
                try:
                    source = create_segment_source(resolved, resume, ctx)
                except UnsupportedSourceError as e:
                    self._unsupported(str(e))
                    continue
                if (
                    assembler is None
                    or assembler.clock_track != source.clock_track
                    or assembler.unwrapper.wrap_period != source.wrap_period
                ):
                    if assembler is not None:
                        await self._flush(info, assembler)
                        self._anchor = None  # a different clock: nothing to continue
                    assembler = self._new_assembler(source)
                self._vod_offset = source.vod_offset

                chunks_before = self._stats.chunks_produced
                current = source.resolved
                try:
                    outcome = await self._consume(info, source, assembler)
                finally:
                    resume = source.checkpoint
                    self._stats.segments_missing += source.segments_missing
                    self._stats.segments_skipped += source.segments_skipped
                    await source.aclose()
                    source = None

                if outcome.kind == "end":
                    end_reason = outcome.reason
                    break
                if outcome.kind == "swap":
                    # Same position, same clock: continue seamlessly on the new URLs.
                    self._event(outcome.reason)
                    pending = outcome.resolved
                    continue
                if outcome.kind == "reset":
                    # A new timeline: new clock state, and the new numbering read from its
                    # start on the same URLs (no yt-dlp run). The source clamps sequence 0 to
                    # the catch-up window or the oldest segment still available.
                    self._event(f"the stream's numbering restarted ({outcome.reason}); following the new one", logging.WARNING)
                    await self._flush(info, assembler)
                    assembler = None
                    self._anchor = None
                    resume = 0
                    pending = current
                    self._stats.reconnects += 1
                    continue
                self._stats.reconnects += 1
                self._pace_after_connection(productive=self._stats.chunks_produced > chunks_before)
                self._event(f"reconnecting: {outcome.reason}")
        except _EndRun as e:
            end_reason = e.reason
        except asyncio.CancelledError:
            # Cleanup only: no final chunk (its emit would block a task being torn down).
            cancelled = True
            end_reason = "cancelled"
            raise
        except Exception as e:
            # Never let an unexpected error kill the watcher.
            logger.exception("[%s] timeline: unexpected error: %s", key, e)
            end_reason = f"unexpected error: {e}"
        finally:
            try:
                if source is not None:
                    with contextlib.suppress(Exception):
                        await source.aclose()
                if assembler is not None and not cancelled:
                    try:
                        await self._flush(info, assembler)
                    except asyncio.CancelledError:
                        cancelled = True
                        end_reason = "cancelled"
                        raise
                    except Exception:
                        logger.exception("[%s] timeline: could not flush the last chunk", key)
            finally:
                try:
                    self._finish(end_reason, cancelled)
                except Exception:
                    logger.exception("[%s] timeline: could not finish the run cleanly", key)
                self._remove_work_dir()
        return self._stats

    async def _consume(self, info: StreamInfo, source: SegmentSource, assembler: ChunkAssembler) -> _Outcome:
        """Feeds one connection's segments to the assembler until it should be
        rebuilt, swapped for fresh URLs, reset, or the run should end."""
        start = self._monotonic()
        last_segment = start
        last_chunk = start
        next_refresh_try = 0.0
        refresh: asyncio.Task[ResolvedStream] | None = None
        errors = 0
        try:
            while not self._should_stop():
                expires_at = source.expires_at
                if expires_at is not None:
                    remaining = expires_at - self._wallclock()
                    if remaining < self._HARD_REFRESH_BEFORE_EXPIRY_SECONDS:
                        return _Outcome("reconnect", "the stream URLs are about to expire")
                    if refresh is not None and refresh.done():
                        refreshed = self._refresh_result(refresh)
                        refresh = None
                        if refreshed is not None:
                            return _Outcome("swap", "refreshed the stream URLs before they expire", refreshed)
                    now = self._monotonic()
                    if (
                        refresh is None
                        and remaining < self._REFRESH_BEFORE_EXPIRY_SECONDS
                        and now >= max(next_refresh_try, self._next_resolve_at)
                    ):
                        # Resolved in the background: the current URLs keep working meanwhile.
                        next_refresh_try = now + self._PROACTIVE_RETRY_SECONDS
                        self._next_resolve_at = now + self._MIN_RESOLVE_INTERVAL_SECONDS
                        refresh = asyncio.create_task(
                            self._resolve_fresh(source.ctx.watch_url), name=f"timeline-{self.streamer.key}-refresh"
                        )

                try:
                    group = await source.next_group(self._POLL_TIMEOUT_SECONDS)
                except SourceResetError as e:
                    return _Outcome("reset", str(e))
                except SourceExpiredError as e:
                    return _Outcome("reconnect", str(e))
                except SourceError as e:
                    errors += 1
                    self._event(f"source error {errors}/{self._MAX_SOURCE_ERRORS}: {e}", logging.WARNING)
                    if errors >= self._MAX_SOURCE_ERRORS:
                        return _Outcome("reconnect", f"repeated source errors ({e})")
                    await self._sleep(min(10.0, 2.0**errors))
                    continue
                errors = 0

                if source.ended:
                    return _Outcome("end", "the stream ended")

                now = self._monotonic()
                if group is None:
                    if now - last_segment > self._IDLE_RECONNECT_SECONDS:
                        return _Outcome("reconnect", f"no new segment in {now - last_segment:.0f}s")
                    continue

                last_segment = now
                self._track_backlog(source)
                for chunk in assembler.add(group):
                    if await self._emit(info, chunk):
                        last_chunk = self._monotonic()
                        # emit() waits while the pipeline is backed up: that wait isn't the
                        # connection going idle.
                        last_segment = last_chunk

                if now - last_chunk > self.config.server.stale_threshold.no_audio_seconds:
                    return _Outcome("reconnect", f"nothing usable in {now - last_chunk:.0f}s")
                for name, missing in source.consecutive_missing.items():
                    if name == source.clock_track or missing < self._TRACK_RECONNECT_AFTER:
                        continue
                    if now - self._last_track_reconnect >= self._TRACK_RECONNECT_INTERVAL_SECONDS:
                        self._last_track_reconnect = now
                        return _Outcome("reconnect", f"no {name} for {missing} segments")
            return _Outcome("end", "stopped")
        finally:
            if refresh is not None:
                # Nobody will use it now: stop its yt-dlp rather than leave it running.
                await self._discard(refresh)

    # ------------------------------------------------------------------
    # Resolution (paced)
    # ------------------------------------------------------------------

    async def _resolve(self, ctx: SourceContext) -> ResolvedStream:
        """Resolves the stream's formats, retrying with backoff while the stream
        is live. Raises _EndRun when it isn't (or on stop)."""
        while True:
            if self._should_stop():
                raise _EndRun("stopped")
            wait = self._next_resolve_at - self._monotonic()
            if wait > 0:
                if wait > self._MIN_RESOLVE_INTERVAL_SECONDS:
                    self._event(f"waiting {wait:.0f}s before asking yt-dlp again")
                await self._sleep(wait)
                continue
            self._next_resolve_at = self._monotonic() + self._MIN_RESOLVE_INTERVAL_SECONDS
            try:
                resolved = await resolve_stream(
                    self.config, ctx.watch_url, self.streamer.media_type, self._should_stop, raw_log=self._run_log.append
                )
            except ResolveStopped as e:
                raise _EndRun("stopped") from e
            except StreamNotLiveError as e:
                self._event(f"the stream is not live: {e}")
                raise _EndRun("the stream is not live") from e
            except StreamUnavailableError as e:
                self._event(f"the stream is not available: {e}", logging.WARNING)
                raise _EndRun(f"the stream is not available ({e})") from e
            except UnsupportedSourceError as e:
                self._unsupported(str(e))
                continue
            except ResolveError as e:
                self._event(f"yt-dlp failed: {e}", logging.WARNING)
                self._backoff()
                continue
            self._unsupported_reason = None
            resolved = self._checked_expiry(resolved)
            formats = ", ".join(f"{t.track}={t.format_id}" for t in resolved.tracks)
            expiry = f", URLs valid for {(resolved.expires_at - self._wallclock()) / 3600:.1f}h" if resolved.expires_at else ""
            self._event(f"resolved {formats}{expiry}")
            return resolved

    async def _resolve_fresh(self, url: str) -> ResolvedStream:
        """Runs as the background refresh task."""
        return await resolve_stream(self.config, url, self.streamer.media_type, self._should_stop, raw_log=self._run_log.append)

    def _refresh_result(self, task: asyncio.Task[ResolvedStream]) -> ResolvedStream | None:
        """The outcome of a background URL refresh; None keeps the current URLs."""
        if task.cancelled():
            return None
        try:
            resolved = task.result()
        except ResolveStopped:
            return None
        except ResolveError as e:
            # Keep using the current URLs; a stream that really ended stops advancing and is caught there.
            self._event(f"refreshing the URLs failed ({e}); keeping the current ones for now", logging.WARNING)
            return None
        except Exception as e:
            logger.exception("[%s] timeline: URL refresh failed: %s", self.streamer.key, e)
            return None
        return self._checked_expiry(resolved)

    @staticmethod
    async def _discard(task: asyncio.Task[ResolvedStream]) -> None:
        """Cancels a background refresh and waits for it (resolve_stream kills
        yt-dlp on cancellation). Its outcome no longer matters."""
        if not task.done():
            task.cancel()
            await asyncio.wait({task})
        if not task.cancelled():
            task.exception()  # retrieved: a failure nobody waits for isn't an error

    def _checked_expiry(self, resolved: ResolvedStream) -> ResolvedStream:
        if resolved.expires_at is None:
            return resolved
        headroom = resolved.expires_at - self._wallclock()
        if headroom < self._MIN_EXPIRY_HEADROOM_SECONDS:
            self._event(
                f"the URLs claim to expire in {headroom:.0f}s; ignoring that and relying on rejected requests instead", logging.WARNING
            )
            return dataclasses.replace(resolved, expires_at=None)
        return resolved

    def _backoff(self) -> None:
        delay = self._BACKOFF_SECONDS[min(self._backoff_step, len(self._BACKOFF_SECONDS) - 1)]
        delay = min(delay, max(self._MIN_RESOLVE_INTERVAL_SECONDS, self._lfs_gap_seconds / 2))
        self._backoff_step += 1
        self._next_resolve_at = max(self._next_resolve_at, self._monotonic() + delay)

    def _pace_after_connection(self, productive: bool) -> None:
        if productive:
            self._backoff_step = 0
        else:
            self._backoff()

    def _unsupported(self, reason: str) -> None:
        if reason != self._unsupported_reason:
            self._event(f"cannot read this stream: {reason}. Retrying every {self._UNSUPPORTED_RETRY_SECONDS / 60:.0f} min.", logging.ERROR)
            self._unsupported_reason = reason
        self._next_resolve_at = max(self._next_resolve_at, self._monotonic() + self._UNSUPPORTED_RETRY_SECONDS)

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def _new_assembler(self, source: SegmentSource) -> ChunkAssembler:
        # Seed the clock's wrap count from the time since the stream started, so
        # a run that begins after the 33-bit clock wrapped lands on the right side.
        expected = None if self._start_time is None else self._wallclock() - self._start_time
        unwrapper = ClockUnwrapper(source.wrap_period, expected)
        return ChunkAssembler(self.config.server.buffer_size_seconds, source.clock_track, unwrapper, on_gap=self._on_gap)

    async def _emit(self, info: StreamInfo, chunk: TimelineChunk) -> bool:
        """Muxes a chunk to a file, decodes its audio and hands it to the
        pipeline. False if it produced nothing usable.

        The file is this capture's until emit() returns: on any failure, and
        when emit() itself is cancelled (an asyncio.Queue.put that raises
        never took the item), it is deleted here."""
        key = self.streamer.key
        ffmpeg = self.config.capture.ffmpeg_path
        sequences = f"{chunk.first_sequence}-{chunk.last_sequence}"
        path = self._work_dir / f"chunk{self._chunk_number:06d}.ts"
        self._chunk_number += 1
        handed_over = False
        try:
            try:
                await mux_chunk(ffmpeg, chunk, self._work_dir / self._MUX_SUBDIR, path)
            except MuxError as e:
                self._stats.chunks_failed += 1
                self._warn("mux", f"mux failed for sequences {sequences}: {e}")
                return False
            except Exception as e:
                self._stats.chunks_failed += 1
                logger.exception("[%s] timeline: mux failed unexpectedly: %s", key, e)
                return False
            try:
                pcm = await audio.extract_pcm(ffmpeg, path)
            except audio.AudioExtractError as e:
                self._stats.chunks_failed += 1
                self._warn("decode", f"sequences {sequences} have no decodable audio ({e})")
                return False
            except Exception as e:
                self._stats.chunks_failed += 1
                logger.exception("[%s] timeline: decoding failed unexpectedly: %s", key, e)
                return False
            duration = audio.pcm_duration(pcm)
            if duration <= 0:
                self._stats.chunks_failed += 1
                self._warn("decode", f"sequences {sequences} have no decodable audio")
                return False
            if duration < self.config.transcription.min_chunk_seconds:
                # Too short to be a line (the pipeline's rule, as in live-edge capture).
                self._stats.chunks_failed += 1
                self._warn("short", f"sequences {sequences} decode to only {duration:.2f}s of audio, under transcription.min_chunk_seconds")
                return False

            now = self._wallclock()
            lag_from: float | None = None
            if self._vod_accurate(chunk, now):
                audio_start_time = self._stream_start_time(chunk, now) + chunk.clock + self._vod_offset
                # An estimated start time follows the clock, but the server measures lines
                # from its activation time instead (see the module docstring).
                vod_accurate = not self._start_time_estimated
                # A clock stamp skips the time the stream was down: the pipeline measures
                # this line's lag from now instead (models.Chunk.lag_from).
                lag_from = time.time()
            else:
                # The clock contradicts the wall clock: stamp by arrival, like live-edge capture.
                audio_start_time = now - duration - self.config.capture.live_latency_seconds
                vod_accurate = False
            if not vod_accurate:
                self._stats.chunks_not_vod_accurate += 1

            if self._stats.chunks_produced == 0:
                tracks = ", ".join(f"{name}={track.container}" for name, track in chunk.tracks.items() if track.data)
                logger.info("[%s] timeline: first chunk ready: %s, %.2fs, stream clock %.3fs", key, tracks, duration, chunk.clock)
            logger.debug("[%s] timeline: queuing sequences %s. Clock: %.3fs, duration: %.3fs", key, sequences, chunk.clock, duration)
            waiting_since = self._monotonic()
            await self._emit_chunk(
                Chunk(
                    key=key,
                    stream_id=info.stream_id,
                    path=path,
                    audio_start_time=audio_start_time,
                    duration=duration,
                    vod_accurate=vod_accurate,
                    media_type=self.streamer.media_type,
                    pcm=pcm,
                    lag_from=lag_from,
                )
            )
            handed_over = True
            # Applied at once: a flush's wait (a numbering reset, a changed clock track) must
            # widen the window before the next source's first lag check.
            self._emit_wait += self._monotonic() - waiting_since
            self._update_catch_up_window()
        finally:
            if not handed_over:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
        self._stats.chunks_produced += 1
        self._save_state(info.stream_id, chunk.last_sequence + 1)
        return True

    def _track_backlog(self, source: SegmentSource) -> None:
        """Keeps the pipeline's backpressure from costing segments.

        emit() waits while the pipeline is backed up (a slow provider, the server
        down), and the source isn't read meanwhile, so it falls behind live by about
        that long. Its catch-up window (lfs_gap_seconds, past which it skips to the
        live edge) is meant for hiccups and restarts; segments this worker's own
        pipeline held up are still being served, so the window is widened by the
        time spent waiting (_catch_up_window) until the source is back within
        lfs_gap_seconds. YouTube's limit still holds: once three older segments in a
        row 404 (gone, or behind a stream break or a change of YouTube's segment
        URLs), the source continues at the oldest one its last playlist read
        listed, near the live edge, as after any outage.

        Called with the position just read, before this round's emits: the wait
        they add is measured from here.
        """
        behind = source.behind_live
        if behind is None:
            return
        self._behind_live = behind
        skipped = self._stats.segments_skipped + source.segments_skipped
        if behind <= self._lfs_gap_seconds:
            if self._backlog_logged:
                if skipped > self._skipped_at_backlog:
                    self._event(f"the pipeline backlog is over, but {skipped - self._skipped_at_backlog} segments were skipped")
                else:
                    self._event(f"caught up with the pipeline backlog ({behind:.0f}s behind live)")
            self._emit_wait = 0.0
            self._backlog_logged = False
            self._update_catch_up_window()
        elif self._emit_wait >= self._BACKLOG_LOG_WAIT_SECONDS and not self._backlog_logged:
            # Only a real wait: a put into a queue with room still takes microseconds, and
            # falling behind for any other reason (slow downloads) isn't the pipeline's doing.
            self._backlog_logged = True
            self._skipped_at_backlog = skipped
            self._event(
                f"the pipeline is backed up and the stream is {behind:.0f}s ahead of the capture; catching up "
                "instead of skipping to the live edge",
                logging.WARNING,
            )

    async def _flush(self, info: StreamInfo, assembler: ChunkAssembler) -> None:
        chunk = assembler.flush()
        if chunk is not None and chunk.duration >= self.config.transcription.min_chunk_seconds:
            await self._emit(info, chunk)

    def _on_gap(self, reason: str) -> None:
        self._stats.gaps += 1
        self._event(f"gap: {reason}")

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_start_time(value: float | str | None) -> float | None:
        if value is None:
            return None
        try:
            start = float(value)
        except (TypeError, ValueError):
            return None
        return start if start > 0 else None

    def _stream_start_time(self, chunk: TimelineChunk, now: float) -> float:
        """The epoch of clock zero as the server sees it: the stream's start time
        (it subtracts that to get a line's VOD position). Estimated once, from
        this chunk's arrival, when the start time is unknown. A run that resumes
        behind live estimates from a catch-up chunk, so the stream published
        since that chunk (behind_live) is counted too: without it every later
        line would land that far in the future."""
        if self._start_time is None:
            self._start_time = now - (chunk.clock + chunk.duration) - self._behind_live - self._ASSUMED_DELIVERY_SECONDS
            self._start_time_estimated = True
            self._event(f"estimated the stream start as {self._start_time:.0f}", logging.WARNING)
        return self._start_time

    def _vod_accurate(self, chunk: TimelineChunk, now: float) -> bool:
        """Whether a chunk's clock stamps VOD-accurate lines. A chunk that continues
        the timeline of the last VOD-accurate one stays VOD-accurate. That matters
        after outages: YouTube's clock (like the VOD) skips the time a stream was
        down, so start time + clock falls behind now by their total length, which
        the start-time check alone would sooner or later take for a restarted
        timeline (seen live: one 8.5 min outage left it 506 s behind)."""
        if self._continues_anchor(chunk, now) or self._clock_trusted(chunk, now):
            self._anchor = (chunk.clock + chunk.duration, now, False)
            return True
        self._anchor = None
        return False

    def _continues_anchor(self, chunk: TimelineChunk, now: float) -> bool:
        if self._anchor is None:
            return False
        clock_end, queued_at, from_state = self._anchor
        advance = chunk.clock - clock_end
        if from_state:
            # Another run's: only an exact continuation proves it's the same timeline.
            return abs(advance) <= self._CONTINUITY_TOLERANCE_SECONDS
        # The clock carries on where it stopped (an outage pauses it) or further on
        # (segments skipped), but never back, and never further than the time since
        # plus the backlog that was being caught up.
        return (
            -self._CONTINUITY_TOLERANCE_SECONDS
            <= advance
            <= (now - queued_at + self._catch_up_window + self._CLOCK_BEHIND_TOLERANCE_SECONDS)
        )

    def _clock_trusted(self, chunk: TimelineChunk, now: float) -> bool:
        """Whether the stream's clock counts from the stream's start, i.e. start
        time + clock lands near now (minus delivery delay and any backlog being
        caught up). YouTube restarts a stream's timeline after some breaks (seen
        on 24/7 streams: sequence numbers and clock reset days in); its clock then
        says nothing about the VOD position."""
        if self._start_time is None or self._start_time_estimated:
            return True  # the start time is derived from the clock itself
        offset = self._start_time + chunk.clock + chunk.duration - now
        if -(self._catch_up_window + self._CLOCK_BEHIND_TOLERANCE_SECONDS) <= offset <= self._CLOCK_AHEAD_TOLERANCE_SECONDS:
            return True
        self._warn(
            "clock",
            f"the stream clock puts this chunk {offset:+.0f}s from now, so it doesn't count from the stream's start "
            "(the stream's timeline restarted?); lines are stamped by arrival time and not marked VOD-accurate",
        )
        return False

    # ------------------------------------------------------------------
    # Resume state
    # ------------------------------------------------------------------

    def _load_state(self, stream_id: str) -> int | None:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        sequence = data.get("next_sequence")
        if data.get("stream_id") != stream_id or not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            return None
        anchor = data.get("anchor")
        if (
            isinstance(anchor, list)
            and len(anchor) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in anchor)
        ):
            self._anchor = (float(anchor[0]), float(anchor[1]), True)
        self._event(f"resuming this stream from sequence {sequence}")
        return sequence

    def _save_state(self, stream_id: str, next_sequence: int) -> None:
        state: dict = {"stream_id": stream_id, "next_sequence": next_sequence}
        # An estimated start time is derived from the clock, so this run never checked the
        # clock against a real start: a later run must not continue its timeline on trust.
        if self._anchor is not None and not self._start_time_estimated:
            state["anchor"] = list(self._anchor[:2])
        try:
            atomic_write(self._state_path, json.dumps(state).encode())
        except OSError as e:
            self._warn("state", f"cannot save the resume point: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _event(self, message: str, level: int = logging.INFO) -> None:
        """Logs to the app log and the per-channel run log."""
        logger.log(level, "[%s] timeline: %s", self.streamer.key, message)
        self._run_log.append(f"{time.strftime('%H:%M:%S')} {message}")

    def _warn(self, kind: str, message: str) -> None:
        """A warning that may repeat every chunk: logged at most every few minutes per kind."""
        now = self._monotonic()
        if now - self._warned_at.get(kind, -1e9) >= self._WARN_INTERVAL_SECONDS:
            self._warned_at[kind] = now
            self._event(message, logging.WARNING)

    async def _sleep(self, seconds: float) -> None:
        end = self._monotonic() + seconds
        while not self._should_stop():
            remaining = end - self._monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.2, remaining))

    def _finish(self, reason: str, cancelled: bool) -> None:
        key = self.streamer.key
        s = self._stats
        summary = (
            f"chunks={s.chunks_produced}, failed={s.chunks_failed}, not VOD-accurate={s.chunks_not_vod_accurate}, "
            f"gaps={s.gaps}, missing segments={s.segments_missing}, skipped segments={s.segments_skipped}, "
            f"reconnects={s.reconnects}"
        )
        if cancelled:
            logger.info("[%s] timeline: cancelled (%s).", key, summary)
        elif self._should_stop():
            logger.info("[%s] timeline: stopped on request (%s).", key, summary)
        elif s.chunks_produced == 0:
            logger.warning("[%s] timeline: run produced no chunks: %s (%s).", key, reason, summary)
        else:
            logger.info("[%s] timeline: run finished: %s (%s).", key, reason, summary)
        self._run_log.append(f"--- {self._NAME} stopped at {time.strftime('%Y-%m-%d %H:%M:%S')}: {reason} ({summary}) ---")

    def _remove_work_dir(self) -> None:
        """Removes the mux scratch dir, and the work dir once it's empty. Chunk
        files still in it travel through the pipeline by path; the pipeline
        deletes (or queues) each one, and an emptied work dir left behind is
        swept at the next worker boot, as live-edge capture's are."""
        shutil.rmtree(self._work_dir / self._MUX_SUBDIR, ignore_errors=True)
        with contextlib.suppress(OSError):
            self._work_dir.rmdir()
