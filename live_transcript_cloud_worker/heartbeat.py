"""Worker heartbeat: POST /status every 60 s, from a task independent of
capture, so a stalled download can never make the worker look offline
(docs/01: the server alerts past 5 minutes of silence)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os

logger = logging.getLogger(__name__)

INTERVAL_SECONDS = 60.0


async def heartbeat_loop(server, keys: list[str], stop_event: asyncio.Event) -> None:
    version = os.environ.get("APP_VERSION", "local")
    build_time = os.environ.get("BUILD_DATE", "unknown")
    while not stop_event.is_set():
        await server.post_status(version, build_time, keys)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=INTERVAL_SECONDS)
