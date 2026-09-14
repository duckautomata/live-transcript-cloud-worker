"""Per-channel watcher: stream discovery, the activate -> capture ->
deactivate cycle, and restart-signal handling (docs/02).

Each watcher owns its channel's durable state and line IDs; nothing else
touches them. URL polling is adaptive per URL, jittered base interval,
scheduled-start awareness, and a long back-off for confirmed-offline URLs,
so N URLs never sync up into probe bursts the platforms rate-limit.

Verdicts come in three grades and the queue rules key on them:

- **trusted**: LIVE, UPCOMING, or a confirmed offline (yt-dlp's "not
  currently live", a resolved was_live/post_live/not_live) seen while the
  YouTube jar authenticates. Only these move the offline-delete counter.
- **inconclusive**: yt-dlp failed for some other reason (a private or removed
  video, a bot check), resolved without a usable live_status, printed no
  metadata (a members-only entry the match filter rejected), or produced
  any OFFLINE verdict, confirmed or not, while the jar is dead. These count
  toward a bounded give-up in incoming mode (at least
  inconclusive_delete_threshold consecutive probes spanning at least
  inconclusive_min_span_seconds) so a queue entry that can never be resolved
  does not stay forever; while the jar is dead the URL is also parked on a
  slow cadence and re-probed the moment cookies.txt is overwritten.
- **error**: timeouts, a missing binary, garbage output, and yt-dlp failures
  on this worker's side (network, rate limit, broken extractor). About this
  worker, not the URL: they neither advance nor reset anything.
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
from .ytdlp import Platform, ProbeOutcome, Prober, platform_of

logger = logging.getLogger(__name__)

TICK_SECONDS = 1.0
POST_CAPTURE_RECHECK_SECONDS = 5.0
PIPELINE_DRAIN_SHUTDOWN_SECONDS = 45.0

# A failed /incoming refresh is retried on this backoff rather than waiting out
# the full fallback interval (600 s in the deployed config), which would leave
# the channel un-probed for ten minutes after a few-second server restart.
INCOMING_RETRY_BASE_SECONDS = 5.0
INCOMING_RETRY_CAP_SECONDS = 120.0


class _UrlSchedule:
    __slots__ = ("next_check", "offline_count", "inconclusive_count", "inconclusive_since", "parked", "woken")

    def __init__(self) -> None:
        self.next_check = 0.0
        # Consecutive trusted confirmed-offline verdicts: the incoming-mode
        # delete threshold. Frozen, not reset, by inconclusive probes.
        self.offline_count = 0
        # Consecutive probes that said nothing about the stream (see the
        # module docstring), and when that streak began. ERROR outcomes
        # neither add nor reset.
        self.inconclusive_count = 0
        self.inconclusive_since = 0.0
        # Set while the URL sits on the degraded-cookie cadence. Cleared by
        # any scheduling with a healthy jar and by a jar wake-up, so each
        # parked episode logs its "parking" line again.
        self.parked = False
        # Set by a jar wake-up until the next probe: that probe never counts
        # toward the give-up (whatever the jar's state by then), so a wake
        # costs one probe round and nothing more.
        self.woken = False

    def conclusive(self) -> None:
        """A trusted verdict arrived: the URL is resolvable again."""
        self.inconclusive_count = 0
        self.parked = False
        self.woken = False

    def note_inconclusive(self, now: float) -> None:
        if self.inconclusive_count == 0:
            self.inconclusive_since = now
        self.inconclusive_count += 1


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
        self._incoming_failures = 0
        # A stream whose deactivate never landed: the server still shows the
        # channel live, so the next chance (URL removal, restart, shutdown)
        # retries it.
        self._pending_deactivate = ""
        # Last cookies.txt generation seen (see _wake_parked_if_jar_replaced).
        self._jar_generation = getattr(prober, "jar_generation", 0)
        if not self._incoming_mode:
            for url in streamer.urls:
                self._schedules[url] = _UrlSchedule()

    # ------------------------------------------------------------- helpers

    def _should_stop(self) -> bool:
        return self.stop_event.is_set() or self.restart_event.is_set()

    def _jittered_interval(self) -> float:
        return self.config.server.channel_polling.interval_seconds + random.uniform(-5, 10)

    def _cookies_degraded_for(self, url: str) -> bool:
        """True when this URL's offline verdict cannot be trusted.

        Twitch never gets cookies, so its verdicts stay authoritative no matter
        what the YouTube jar is doing.
        """
        if platform_of(url) is Platform.TWITCH:
            return False
        # Read through the prober so watchers and probes can never disagree
        # about the one worker-global jar. A watcher built without a prober
        # (scheduling tests) has no cookie state and is treated as healthy.
        cookies = getattr(self.prober, "cookies", None)
        return cookies is not None and cookies.degraded

    def _schedule_after_probe(self, schedule: _UrlSchedule, outcome: ProbeOutcome, url: str) -> None:
        polling = self.config.server.channel_polling
        now = time.time()
        woken, schedule.woken = schedule.woken, False
        if outcome.result is ProbeResult.UPCOMING:
            # A positive verdict: the broadcast exists and has not started. A
            # late streamer ("will begin in a few moments") carries no start
            # time and simply stays on the base cadence.
            schedule.offline_count = 0
            schedule.conclusive()
            if outcome.scheduled_start:
                wake = outcome.scheduled_start - polling.pre_scheduled_buffer_seconds
                schedule.next_check = max(
                    now + self._jittered_interval(),
                    min(wake, now + polling.max_interval_seconds),
                )
            else:
                schedule.next_check = now + self._jittered_interval()
            return
        if self._cookies_degraded_for(url):
            self._park(schedule, outcome, url, now, woken)
            return
        # Scheduled with a healthy jar: off the degraded cadence, and the
        # next degraded episode logs its "parking" line again.
        schedule.parked = False
        if outcome.result is ProbeResult.OFFLINE and outcome.confirmed_offline:
            schedule.conclusive()
            schedule.offline_count += 1
            if not self._incoming_mode and platform_of(url) is not Platform.TWITCH:
                schedule.next_check = now + polling.max_interval_seconds
            else:
                # Twitch streams start unannounced, the long confirmed-offline
                # back-off would miss them; keep the base cadence. Incoming
                # mode also stays at base cadence so the delete threshold can
                # accumulate promptly.
                schedule.next_check = now + self._jittered_interval()
            return
        if outcome.inconclusive and not woken:
            # A private/removed video, a bot check, an unknown live_status:
            # nothing here says the stream is over, so the offline streak is
            # left alone, but nothing says it is resolvable either, and in
            # incoming mode the give-up threshold bounds how long the queue
            # entry can hang on a verdict that never comes.
            schedule.note_inconclusive(now)
        # ERROR (timeout, missing binary, garbage output, a failure on this
        # worker's side) neither advances nor resets any counter.
        schedule.next_check = now + self._jittered_interval()

    def _park(self, schedule: _UrlSchedule, outcome: ProbeOutcome, url: str, now: float, woken: bool) -> None:
        """Signed out, every YouTube verdict is suspect: a bot-checked or
        members-only stream reads as "Sign in to confirm you're not a bot"
        whether or not it is live, and a /channel/<id>/live URL whose stream
        we cannot see raises yt-dlp's UserNotLive -- the very "not currently
        live" text a genuine offline produces. So while the jar is dead no
        verdict counts as offline (the offline streak is frozen, not reset,
        and resumes once cookies are back), and the URL moves to the slow
        degraded cadence: every probe with a dead jar is a wasted yt-dlp run
        that also rewrites cookies.txt. An OFFLINE verdict, confirmed or not,
        counts as inconclusive so the incoming-mode give-up eventually bounds
        the outage instead of holding the queue entry until the server's
        TTL; an ERROR is parked on the same cadence without counting; LIVE
        and UPCOMING are positive verdicts a signed-out probe can still be
        trusted on and never reach here. A probe triggered by a jar wake-up
        never counts either, so a false wake (a download's exit-time
        rewrite) cannot shorten the documented bound. An overwritten
        cookies.txt wakes parked URLs at once (_wake_parked_if_jar_replaced),
        so recovery does not wait out the cadence.
        """
        if outcome.result is ProbeResult.OFFLINE and not woken:
            schedule.note_inconclusive(now)
        interval = self.config.server.cookies.degraded_probe_interval_seconds
        if not schedule.parked:
            schedule.parked = True
            cookies = self.prober.cookies
            logger.warning(
                "[%s] parking %s while youtube cookies are %s (%s): probing every %.0fs; an in-place overwrite of "
                "cookies.txt re-probes it at once, and if this line follows one the new jar is not authenticating "
                "either (last probe: %s)",
                self.streamer.key,
                url,
                cookies.state.value,
                cookies.reason or "not authenticating",
                interval,
                outcome.reason or outcome.result.value,
                extra={"url": url, "cookie_state": cookies.state.value, "inconclusive_count": schedule.inconclusive_count},
            )
        schedule.next_check = now + interval + random.uniform(0, 10)

    def _wake_parked_if_jar_replaced(self) -> None:
        """Re-probe parked URLs the moment the operator overwrites the jar."""
        poll = getattr(self.prober, "poll_jar", None)
        if poll is None:
            return
        generation = poll()
        if generation == self._jar_generation:
            return
        self._jar_generation = generation
        parked = [url for url, schedule in self._schedules.items() if schedule.parked]
        if not parked:
            return
        logger.info("[%s] cookies.txt changed; re-probing %d parked URL(s) now", self.streamer.key, len(parked))
        for url in parked:
            schedule = self._schedules[url]
            schedule.next_check = 0.0
            # Un-park so a re-park with the new jar logs again (a swapped-in
            # jar that is still dead must not go back to the slow cadence
            # silently), and mark the probe as wake-triggered so it cannot
            # count toward the give-up.
            schedule.parked = False
            schedule.woken = True

    def _give_up_due(self, schedule: _UrlSchedule, now: float) -> bool:
        """The incoming-mode give-up: at least ``inconclusive_delete_threshold``
        consecutive inconclusive probes AND a streak at least
        ``inconclusive_min_span_seconds`` long. The count guarantees real
        attempts on a slow cadence; the span keeps a short worker-wide
        hiccup (a datacenter bot check with a healthy jar, an outage the
        error grade did not recognise) from draining every channel's queue
        in ten minutes. 0 disables the respective condition. The caller only
        consults this after an OFFLINE probe, so the removal is never
        triggered, or logged, by an ERROR.
        """
        incoming = self.config.server.incoming_polling
        if incoming.inconclusive_delete_threshold <= 0 or schedule.inconclusive_count < incoming.inconclusive_delete_threshold:
            return False
        return now - schedule.inconclusive_since >= incoming.inconclusive_min_span_seconds

    # --------------------------------------------------------- deactivation

    async def _deactivate(self, stream_id: str) -> None:
        """Deactivate one stream and clear the local live flag.

        A failed call is remembered rather than forgotten: a channel left
        marked live on the server never self-heals otherwise.
        """
        if not stream_id:
            return
        ok = await self.server.deactivate(self.streamer.key, stream_id)
        self._pending_deactivate = "" if ok else stream_id
        self.state.set_live(False)

    async def _deactivate_current(self, reason: str) -> None:
        """Deactivate whatever the server may still consider live on this
        channel. A no-op when there is nothing outstanding, so callers can
        use it as a belt-and-braces step without generating traffic."""
        stream_id = self.state.stream_id if self.state.is_live else self._pending_deactivate
        if not stream_id:
            return
        logger.info("[%s] deactivating %s (%s)", self.streamer.key, stream_id, reason)
        await self._deactivate(stream_id)

    # ------------------------------------------------------------- incoming

    async def _refresh_incoming(self) -> None:
        urls = await self.server.get_incoming(self.streamer.key)
        if urls is None:
            # The server did not answer. An unanswered poll says nothing about
            # the queue, so keep what we have: treating it as an empty queue
            # would drop every URL and stop probing the channel entirely.
            self._defer_incoming_retry()
            return
        self._incoming_failures = 0
        self._last_incoming_refresh = time.time()
        known = set(self._schedules)
        current = set(urls)
        for url in current - known:
            logger.info("[%s] incoming URL queued: %s", self.streamer.key, url)
            self._schedules[url] = _UrlSchedule()  # next_check=0 -> immediate
        dropped = known - current
        for url in dropped:
            logger.info("[%s] incoming URL removed server-side: %s", self.streamer.key, url)
            del self._schedules[url]
        if dropped:
            # The queue entry is gone, so nothing will re-capture this
            # channel's stream: make sure the server isn't left showing it
            # live (docs/02). Usually already deactivated -> no request.
            await self._deactivate_current("incoming URL removed server-side")

    def _incoming_interval(self) -> float:
        events = self.config.server.events_polling
        if events.enabled:
            return events.fallback_interval_seconds
        return self.config.server.incoming_polling.interval_seconds

    def _incoming_fallback_elapsed(self) -> bool:
        return time.time() - self._last_incoming_refresh >= self._incoming_interval()

    def _incoming_view_stale(self) -> bool:
        """True when the local URL set may predate the operator's last edit.

        The events long-poll only announces ADDED URLs (the server keys it on
        received_at), so a URL removed on the admin page is otherwise unseen
        until the fallback refresh, up to fallback_interval_seconds of
        probing something nobody wants. A probe costs seconds of yt-dlp CPU;
        a queue read is one cheap GET, so before a probe the view is re-read
        whenever it is older than incoming_polling.interval_seconds. The
        backoff check is load-bearing: _defer_incoming_retry rewinds
        _last_incoming_refresh, and without it a server outage would turn
        every due probe into another failing GET.
        """
        if self._incoming_failures:
            return False
        return time.time() - self._last_incoming_refresh >= self.config.server.incoming_polling.interval_seconds

    def _defer_incoming_retry(self) -> None:
        """Re-arm the fallback timer to fire again after a short backoff.

        The timestamp is moved rather than left alone: an untouched
        ``_last_incoming_refresh`` keeps ``_incoming_fallback_elapsed`` true,
        and the run loop would re-poll on every 1 s tick for the whole outage.
        """
        self._incoming_failures += 1
        delay = min(
            INCOMING_RETRY_CAP_SECONDS,
            INCOMING_RETRY_BASE_SECONDS * 2 ** (self._incoming_failures - 1),
        )
        self._last_incoming_refresh = time.time() - self._incoming_interval() + delay
        logger.warning(
            "[%s] incoming refresh failed; keeping %d known URL(s), retrying in %.0fs",
            self.streamer.key,
            len(self._schedules),
            delay,
        )

    async def _drop_incoming_url(self, url: str, reason: str) -> bool:
        """Remove a queued URL from the server's incoming queue.

        Returns False when the DELETE did not land. The URL then stays
        scheduled with its counters intact, so the next probe of the same
        grade retries the removal; popping it anyway would have the next
        queue refresh re-add it with a clean slate and the whole streak
        would start over.
        """
        logger.info("[%s] removing %s from incoming queue: %s", self.streamer.key, url, reason, extra={"url": url})
        # Deactivate first: if the stream's own deactivate never landed (or
        # activation failed before capture ever ran), this is the last point
        # at which we still know the stream ID.
        await self._deactivate_current("incoming URL removed")
        if not await self.server.delete_incoming(self.streamer.key, url):
            logger.warning(
                "[%s] could not remove %s from incoming queue; retrying on its next probe of the same verdict", self.streamer.key, url
            )
            return False
        self._schedules.pop(url, None)
        return True

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

        await self._deactivate(info.stream_id)
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
        await self._deactivate_current("restart signal")
        # Belt-and-braces ack: the events listener already tried; the signal
        # is level-triggered until a DELETE lands (404 = already cleared).
        await self.server.ack_restart(self.streamer.key)
        if self._incoming_mode:
            # Fetch before clearing. Clearing first and then failing to reach
            # the server would leave the channel with no URLs at all -- an
            # admin-stop that lands during a redeploy would silently disable
            # the channel rather than restarting it.
            urls = await self.server.get_incoming(self.streamer.key)
            if urls is None:
                self._defer_incoming_retry()
            else:
                self._incoming_failures = 0
                self._last_incoming_refresh = time.time()
                self._schedules = {url: _UrlSchedule() for url in urls}
        else:
            # Static URLs come from config; re-probe them all immediately.
            for schedule in self._schedules.values():
                schedule.next_check = 0.0
                schedule.offline_count = 0
                schedule.conclusive()
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

                    self._wake_parked_if_jar_replaced()
                    now = time.time()
                    due = [u for u, s in self._schedules.items() if s.next_check <= now]
                    for url in due:
                        if self._should_stop():
                            break
                        if self._incoming_mode and self._incoming_view_stale():
                            # Never spend a probe on a URL the operator may
                            # already have removed: a URL gone from the
                            # server is gone from _schedules after this, and
                            # _check_url skips it.
                            await self._refresh_incoming()
                        await self._check_url(url)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A watcher must never die silently: the heartbeat would
                    # keep the channel looking healthy while nothing runs.
                    logger.exception("[%s] watcher iteration failed; continuing", self.streamer.key)

                await self._sleep_tick()
        finally:
            await self._deactivate_current("watcher stopping")
        logger.info("[%s] watcher stopped", self.streamer.key)

    async def _check_url(self, url: str) -> None:
        schedule = self._schedules.get(url)
        if schedule is None:
            return
        outcome = await self.prober.probe(url, self.state)
        info = outcome.info

        if outcome.result is ProbeResult.LIVE:
            schedule.offline_count = 0
            schedule.conclusive()
            if info.stream_id in self.config.id_blacklist:
                # Blacklisted streams are skipped entirely and never count
                # toward the offline-delete threshold (docs/06).
                logger.info("[%s] ignoring blacklisted stream %s", self.streamer.key, info.stream_id)
                schedule.next_check = time.time() + self._jittered_interval()
                return
            captured = await self._run_stream(info)
            if captured and not self._should_stop() and self._incoming_mode:
                # Capture ended on its own. When the stream is really over
                # the verify probe says so outright (was_live, UserNotLive)
                # and the queued URL goes now (docs/02) rather than after the
                # offline-delete threshold: the fast path for the common
                # case. Anything short of a trusted verdict -- a capture that
                # died mid-stream (LIVE), a probe timeout while the pipeline
                # drains, a bot check because the jar died during the capture
                # -- falls through to the normal counters instead of deleting
                # the operator's URL on the least trustworthy probe of all.
                verify = await self.prober.probe(url, self.state)
                if verify.confirmed_offline and not self._cookies_degraded_for(url):
                    if await self._drop_incoming_url(url, f"stream ended ({verify.reason or 'confirmed offline'})"):
                        return
                    # The DELETE did not land. The verify verdict was a
                    # trusted offline, so let the very next one retry it
                    # instead of waiting out the whole threshold again.
                    schedule.offline_count = max(schedule.offline_count, self.config.server.incoming_polling.offline_delete_threshold - 1)
            # Re-probe soon: a capture that died mid-stream resumes quickly,
            # and a finished stream reads offline and backs off normally.
            schedule = self._schedules.get(url)
            if schedule is not None:
                schedule.next_check = time.time() + POST_CAPTURE_RECHECK_SECONDS
            return

        self._schedule_after_probe(schedule, outcome, url)
        if not self._incoming_mode:
            return
        incoming = self.config.server.incoming_polling
        if (
            outcome.confirmed_offline
            # Deleting the operator's queued URL is irreversible from here, so
            # the offline streak never advances on a verdict a dead cookie
            # could have produced (see _park).
            and not self._cookies_degraded_for(url)
            and schedule.offline_count >= incoming.offline_delete_threshold
        ):
            await self._drop_incoming_url(url, f"confirmed offline {schedule.offline_count}x")
        elif outcome.result is ProbeResult.OFFLINE and self._give_up_due(schedule, time.time()):
            cookies = getattr(self.prober, "cookies", None)
            cookie_state = cookies.state.value if cookies is not None else "n/a"
            logger.warning(
                "[%s] giving up on %s: %d consecutive probes could not tell whether the stream is live "
                "(last: %s; youtube cookies %s). It may still be live; re-queue it once the cause is fixed.",
                self.streamer.key,
                url,
                schedule.inconclusive_count,
                outcome.reason or outcome.result.value,
                cookie_state,
                extra={"url": url, "cookie_state": cookie_state, "inconclusive_count": schedule.inconclusive_count},
            )
            await self._drop_incoming_url(url, f"inconclusive {schedule.inconclusive_count}x")

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
