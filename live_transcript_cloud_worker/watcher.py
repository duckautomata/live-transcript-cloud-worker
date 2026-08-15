"""Per-channel watcher: stream discovery, the activate -> capture ->
deactivate cycle, and restart-signal handling (docs/02).

Each watcher owns its channel's durable state and line IDs; nothing else
touches them. URL polling is adaptive per URL, jittered base interval,
scheduled-start awareness, and a long back-off for confirmed-offline URLs,
so N URLs never sync up into probe bursts the platforms rate-limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time

from . import capture_dash
from .capture import capture_stream
from .config import Config, StreamerConfig
from .models import ProbeResult, StreamInfo
from .pipeline import StreamPipeline
from .state import ChannelState
from .transcribe import TranscriptionService
from .uploader import MediaUploader
from .ytdlp import Platform, Prober, platform_of

logger = logging.getLogger(__name__)

TICK_SECONDS = 1.0
POST_CAPTURE_RECHECK_SECONDS = 5.0
PIPELINE_DRAIN_SHUTDOWN_SECONDS = 45.0


class _UrlSchedule:
    __slots__ = ("next_check", "offline_count")

    def __init__(self) -> None:
        self.next_check = 0.0
        self.offline_count = 0


class ChannelWatcher:
    def __init__(
        self,
        config: Config,
        streamer: StreamerConfig,
        state: ChannelState,
        server,
        prober: Prober,
        transcriber: TranscriptionService,
        uploader: MediaUploader,
        stop_event: asyncio.Event,
    ) -> None:
        self.config = config
        self.streamer = streamer
        self.state = state
        self.server = server
        self.prober = prober
        self.transcriber = transcriber
        self.uploader = uploader
        self.stop_event = stop_event
        self.restart_event = asyncio.Event()
        self.incoming_event = asyncio.Event()
        self._schedules: dict[str, _UrlSchedule] = {}
        self._incoming_mode = config.server.incoming_polling.enabled
        self._last_incoming_refresh = 0.0
        if not self._incoming_mode:
            for url in streamer.urls:
                self._schedules[url] = _UrlSchedule()

    # ------------------------------------------------------------- helpers

    def _should_stop(self) -> bool:
        return self.stop_event.is_set() or self.restart_event.is_set()

    def _jittered_interval(self) -> float:
        return self.config.server.channel_polling.interval_seconds + random.uniform(-5, 10)

    def _schedule_after_probe(self, schedule: _UrlSchedule, outcome, url: str) -> None:
        polling = self.config.server.channel_polling
        now = time.time()
        if outcome.result is ProbeResult.UPCOMING and outcome.scheduled_start:
            schedule.offline_count = 0
            wake = outcome.scheduled_start - polling.pre_scheduled_buffer_seconds
            schedule.next_check = max(
                now + self._jittered_interval(),
                min(wake, now + polling.max_interval_seconds),
            )
        elif outcome.result is ProbeResult.OFFLINE and outcome.confirmed_offline:
            schedule.offline_count += 1
            if not self._incoming_mode and platform_of(url) is not Platform.TWITCH:
                schedule.next_check = now + polling.max_interval_seconds
            else:
                # Twitch streams start unannounced, the long confirmed-offline
                # back-off would miss them; keep the base cadence. Incoming
                # mode also stays at base cadence so the delete threshold can
                # accumulate promptly.
                schedule.next_check = now + self._jittered_interval()
        else:
            schedule.next_check = now + self._jittered_interval()

    # ------------------------------------------------------------- incoming

    async def _refresh_incoming(self) -> None:
        self._last_incoming_refresh = time.time()
        urls = await self.server.get_incoming(self.streamer.key)
        known = set(self._schedules)
        current = set(urls)
        for url in current - known:
            logger.info("[%s] incoming URL queued: %s", self.streamer.key, url)
            self._schedules[url] = _UrlSchedule()  # next_check=0 -> immediate
        for url in known - current:
            del self._schedules[url]

    def _incoming_fallback_elapsed(self) -> bool:
        events = self.config.server.events_polling
        interval = events.fallback_interval_seconds if events.enabled else self.config.server.incoming_polling.interval_seconds
        return time.time() - self._last_incoming_refresh >= interval

    async def _drop_incoming_url(self, url: str) -> None:
        logger.info("[%s] removing offline URL from incoming queue: %s", self.streamer.key, url)
        await self.server.delete_incoming(self.streamer.key, url)
        self._schedules.pop(url, None)

    # ------------------------------------------------------------- capture

    async def _activate(self, info: StreamInfo) -> bool:
        start_time = info.start_time or int(time.time())
        if self.state.stream_id == info.stream_id:
            # Reconnect to a stream already in progress: keep the transcript
            # and nextLineId, refresh title/startTime (titles change). The
            # mediaType is immutable per stream (docs/01), so the recorded
            # one wins over the config even if the config changed meanwhile.
            media_type = self.state.media_type
            if media_type is not self.streamer.media_type:
                logger.warning(
                    "[%s] config media_type %s differs from stream %s's recorded %s; keeping recorded",
                    self.streamer.key,
                    self.streamer.media_type.value,
                    info.stream_id,
                    media_type.value,
                )
            self.state.refresh_stream(info.title, start_time)
        else:
            media_type = self.streamer.media_type
            self.state.start_new_stream(info.stream_id, info.title, start_time, media_type)
        ok = await self.server.activate(self.streamer.key, info.stream_id, info.title, start_time, media_type)
        if not ok:
            logger.error(
                "[%s] activate failed for %s; not capturing (lines would fail too)",
                self.streamer.key,
                info.stream_id,
            )
        return ok

    async def _run_stream(self, info: StreamInfo) -> bool:
        """Capture one live stream. Returns True when capture actually ran
        (regardless of how it ended), False when activation failed."""
        if not await self._activate(info):
            return False

        pipeline = StreamPipeline(self.config, self.streamer, self.state, self.server, self.transcriber, self.uploader)
        pipeline_task = asyncio.create_task(pipeline.run())
        try:
            await self._capture(info, pipeline.emit)
        finally:
            try:
                await pipeline.finish()
                # Give the backlog a bounded window on shutdown; unbounded
                # waits otherwise (the backlog is small by construction).
                if self.stop_event.is_set():
                    done, _ = await asyncio.wait([pipeline_task], timeout=PIPELINE_DRAIN_SHUTDOWN_SECONDS)
                    if not done:
                        logger.warning(
                            "[%s] pipeline did not drain within shutdown budget; cancelling",
                            self.streamer.key,
                        )
                        pipeline_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pipeline_task
            except asyncio.CancelledError:
                # The watcher itself is being cancelled: the pipeline task
                # must not outlive it unobserved.
                pipeline_task.cancel()
                raise

        await self.server.deactivate(self.streamer.key, info.stream_id)
        self.state.set_live(False)
        logger.info(
            "[%s] stream %s finished with %d lines",
            self.streamer.key,
            info.stream_id,
            len(self.state.transcript),
        )
        return True

    async def _capture(self, info: StreamInfo, emit) -> None:
        """Strategy selection (docs/03): YouTube uses DASH live-from-start
        when ``use_dash_for_youtube`` is set (exact timestamps, captures from
        second zero); everything else, Twitch included, captures at the
        live edge. DASH falls back to live-edge when it fails early or falls
        too far behind."""
        key = self.streamer.key
        use_dash = self.config.server.use_dash_for_youtube and platform_of(info.url) is Platform.YOUTUBE
        if use_dash:
            # Pre-flight gap check: if catching up would take longer than the
            # stream likely has left, skip straight to live-edge (docs/03).
            start = float(info.start_time or time.time())
            resume = capture_dash.resume_point(self.state, info.stream_id, start)
            behind = time.time() - resume
            if behind > self.config.server.stale_threshold.lfs_gap_seconds:
                logger.info(
                    "[%s] DASH resume point is %.0fs behind live (> lfs_gap %.0fs); using live-edge",
                    key,
                    behind,
                    self.config.server.stale_threshold.lfs_gap_seconds,
                )
            else:
                stats = await capture_dash.capture_stream_dash(self.config, self.streamer, self.state, info, emit, self._should_stop)
                if self._should_stop() or not stats.fell_behind:
                    return  # stream ended (or we're stopping) under DASH
                logger.warning("[%s] continuing stream %s with live-edge capture", key, info.stream_id)
        await capture_stream(self.config, self.streamer, self.state, info, emit, self._should_stop)

    # ------------------------------------------------------------- restart

    async def _handle_restart(self) -> None:
        """Capture has already unwound (restart_event is in _should_stop).

        A restart serves two operator intents, and both must work:

        - **Reconnect**: force yt-dlp to re-attach to a stream that's still
          live. Channel state (transcript, nextLineId, DASH resume) is KEPT,
          so the re-detected stream resumes at the next line via the same-ID
          reactivation path, never a restart from line 0.
        - **Stop** (the admin-stop button, which also clears the server's
          incoming queue): in incoming mode the local URL set is rebuilt
          from the server's queue rather than reused, keeping the stale set
          would make the worker instantly re-detect and re-capture the very
          stream the operator just stopped.
        """
        logger.info("[%s] restart signal handled; re-probing everything", self.streamer.key)
        if self.state.stream_id and self.state.is_live:
            await self.server.deactivate(self.streamer.key, self.state.stream_id)
            self.state.set_live(False)
        # Belt-and-braces ack: the events listener already tried; the signal
        # is level-triggered until a DELETE lands (404 = already cleared).
        await self.server.ack_restart(self.streamer.key)
        if self._incoming_mode:
            self._schedules.clear()
            await self._refresh_incoming()
        else:
            # Static URLs come from config; re-probe them all immediately.
            for schedule in self._schedules.values():
                schedule.next_check = 0.0
                schedule.offline_count = 0
        self.restart_event.clear()

    # ------------------------------------------------------------- main loop

    async def run(self) -> None:
        logger.info(
            "[%s] watcher started (%s mode, %d urls)",
            self.streamer.key,
            "incoming" if self._incoming_mode else "static",
            len(self._schedules),
        )
        try:
            while not self.stop_event.is_set():
                try:
                    if self.restart_event.is_set():
                        await self._handle_restart()
                        continue

                    if self._incoming_mode and (self.incoming_event.is_set() or self._incoming_fallback_elapsed()):
                        self.incoming_event.clear()
                        await self._refresh_incoming()

                    now = time.time()
                    due = [u for u, s in self._schedules.items() if s.next_check <= now]
                    for url in due:
                        if self._should_stop():
                            break
                        await self._check_url(url)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A watcher must never die silently: the heartbeat would
                    # keep the channel looking healthy while nothing runs.
                    logger.exception("[%s] watcher iteration failed; continuing", self.streamer.key)

                await self._sleep_tick()
        finally:
            if self.state.stream_id and self.state.is_live:
                await self.server.deactivate(self.streamer.key, self.state.stream_id)
                self.state.set_live(False)
        logger.info("[%s] watcher stopped", self.streamer.key)

    async def _check_url(self, url: str) -> None:
        schedule = self._schedules.get(url)
        if schedule is None:
            return
        outcome = await self.prober.probe(url, self.state)
        info = outcome.info

        if outcome.result is ProbeResult.LIVE:
            if info.stream_id in self.config.id_blacklist:
                # Blacklisted streams are skipped entirely and never count
                # toward the offline-delete threshold (docs/06).
                logger.info("[%s] ignoring blacklisted stream %s", self.streamer.key, info.stream_id)
                schedule.offline_count = 0
                schedule.next_check = time.time() + self._jittered_interval()
                return
            schedule.offline_count = 0
            captured = await self._run_stream(info)
            if captured and not self._should_stop() and self._incoming_mode:
                # Capture ended on its own. If the stream is really over,
                # remove its queued URL now (docs/02), ended YouTube streams
                # read as "was_live", never "not currently live", so the
                # offline-delete threshold would never fire for them. A
                # capture that died mid-stream sees LIVE here and resumes.
                verify = await self.prober.probe(url, self.state)
                if verify.result is not ProbeResult.LIVE:
                    await self._drop_incoming_url(url)
                    return
            # Re-probe soon: a capture that died mid-stream resumes quickly,
            # and a finished stream reads offline and backs off normally.
            schedule = self._schedules.get(url)
            if schedule is not None:
                schedule.next_check = time.time() + POST_CAPTURE_RECHECK_SECONDS
            return

        self._schedule_after_probe(schedule, outcome, url)
        if (
            self._incoming_mode
            and outcome.confirmed_offline
            and schedule.offline_count >= self.config.server.incoming_polling.offline_delete_threshold
        ):
            await self._drop_incoming_url(url)

    async def _sleep_tick(self) -> None:
        """Sleep one tick, waking early for stop/restart/incoming signals."""
        waiters = [
            asyncio.create_task(self.stop_event.wait()),
            asyncio.create_task(self.restart_event.wait()),
        ]
        if self._incoming_mode:
            waiters.append(asyncio.create_task(self.incoming_event.wait()))
        try:
            await asyncio.wait(waiters, timeout=TICK_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waiters:
                w.cancel()
