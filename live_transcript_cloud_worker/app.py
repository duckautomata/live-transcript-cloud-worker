"""Supervisor: wiring, startup validation, signals, ordered shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import signal

from . import app_version, build_time
from .config import Config, check_executables
from .events import EventsListener
from .heartbeat import heartbeat_loop
from .server_client import LocalClient, ServerClient
from .state import ChannelState
from .transcribe import TranscriptionService
from .uploader import MediaUploader
from .watcher import ChannelWatcher
from .ytdlp import Prober

logger = logging.getLogger(__name__)

WATCHER_SHUTDOWN_GRACE = 60.0
UPLOAD_DRAIN_SECONDS = 30.0


async def run_app(config: Config) -> int:
    problems = check_executables(config)
    if problems:
        for p in problems:
            logger.critical(p)
        return 1

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_stop, stop_event, sig)

    streamers = config.active_streamers
    states = {s.key: ChannelState(s.key, config.channel_dir(s.key)) for s in streamers}
    for state in states.values():
        state.clear_segments_dir()

    server = ServerClient(config) if config.server.enabled else LocalClient(config)
    transcriber = TranscriptionService(config.transcription)
    uploader = MediaUploader(config, server)
    prober = Prober(config)

    logger.info(
        "live-transcript-cloud-worker %s (build %s) starting: %d channel(s), provider=%s%s, server=%s",
        app_version(),
        build_time(),
        len(streamers),
        config.transcription.provider,
        f"+{config.transcription.fallback_provider}" if config.transcription.fallback_provider else "",
        config.server.url if config.server.enabled else "disabled (local mode)",
    )
    if config.server.enabled:
        version = await server.server_version()
        if version:
            logger.info("server reports version=%s buildTime=%s", version.get("version"), version.get("buildTime"))
        else:
            logger.warning("could not read server version; check connectivity")

    uploader.start()
    uploader.rescan(states)

    watchers = {s.key: ChannelWatcher(config, s, states[s.key], server, prober, transcriber, uploader, stop_event) for s in streamers}

    background: list[asyncio.Task] = [
        asyncio.create_task(heartbeat_loop(server, list(watchers), stop_event), name="heartbeat"),
    ]
    if config.server.enabled:
        listener = EventsListener(config, server, watchers, stop_event)
        background.append(asyncio.create_task(listener.run(), name="events"))
    for task in background:
        _observe(task)

    watcher_tasks = []
    for key, watcher in watchers.items():
        task = asyncio.create_task(watcher.run(), name=f"watcher:{key}")
        _observe(task)
        watcher_tasks.append(task)
        # Staggered start desynchronises platform polling across channels.
        await asyncio.sleep(random.uniform(0.3, 1.5))

    await stop_event.wait()
    logger.info("shutting down: waiting for captures to unwind and queues to drain")

    done, pending = await asyncio.wait(watcher_tasks, timeout=WATCHER_SHUTDOWN_GRACE)
    for task in pending:
        logger.warning("cancelling slow %s", task.get_name())
        task.cancel()
    for task in pending:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    _log_task_errors(done)

    if not await uploader.drain(UPLOAD_DRAIN_SECONDS):
        logger.warning("%d media upload(s) left on disk; they upload on next boot", uploader.pending_count())
    await uploader.stop()

    for task in background:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # Belt-and-braces: any channel still marked live gets a deactivate.
    for key, state in states.items():
        if state.is_live and state.stream_id:
            await server.deactivate(key, state.stream_id)
            state.set_live(False)

    await transcriber.aclose()
    await server.aclose()
    logger.info("shutdown complete")
    return 0


def _request_stop(stop_event: asyncio.Event, sig: signal.Signals) -> None:
    if stop_event.is_set():
        logger.warning("second %s; exiting hard", sig.name)
        raise SystemExit(130)
    logger.info("received %s; beginning graceful shutdown", sig.name)
    stop_event.set()


def _observe(task: asyncio.Task) -> None:
    """Log a supervised task's death the moment it happens, a crashed
    watcher or listener must not hide behind a healthy heartbeat."""

    def _done(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.critical("%s crashed: %r", t.get_name(), exc, exc_info=exc)

    task.add_done_callback(_done)


def _log_task_errors(tasks) -> None:
    for task in tasks:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            logger.error("%s crashed: %r", task.get_name(), exc, exc_info=exc)
