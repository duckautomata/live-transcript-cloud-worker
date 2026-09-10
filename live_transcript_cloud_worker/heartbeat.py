"""Worker heartbeat: POST /status every 60 s, from a task independent of
capture, so a stalled download can never make the worker look offline
(docs/01: the server alerts past 5 minutes of silence)."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from . import app_version, build_time
from .cookieauth import CookieAuthTracker

logger = logging.getLogger(__name__)

INTERVAL_SECONDS = 60.0


async def heartbeat_loop(
    server,
    keys: list[str],
    stop_event: asyncio.Event,
    cookies: CookieAuthTracker | None = None,
) -> None:
    version = app_version()
    built = build_time()
    while not stop_event.is_set():
        # Cookie health rides the heartbeat rather than getting its own
        # endpoint: it already runs at the right cadence, already covers every
        # key in one request, and it answers "is no cookie report a dead cookie
        # or a dead reporter?" for free -- a missing heartbeat is already an
        # alert of its own.
        await server.post_status(
            version,
            built,
            keys,
            cookie_state=cookies.state.value if cookies else None,
            cookie_reason=cookies.reason if cookies else "",
        )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=INTERVAL_SECONDS)
