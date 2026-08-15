"""The /events long-poll listener, with graceful degradation (docs/01).

One long-poll covers every channel; signals fan out to the watchers:

- ``incoming`` is edge-triggered on the queue's newest received_at, the
  returned cursor MUST be echoed back or it re-fires forever.
- ``restart`` is level-triggered until the DELETE ack lands, so a non-empty
  response is followed by a 1 s sleep to keep a failing ack from becoming a
  hot loop.

Any /events failure degrades that round to interval polling: check
/{channel}/restart per channel, nudge the incoming refresh, sleep, retry.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from .config import Config
from .watcher import ChannelWatcher

logger = logging.getLogger(__name__)


class EventsListener:
    def __init__(
        self,
        config: Config,
        server,
        watchers: dict[str, ChannelWatcher],
        stop_event: asyncio.Event,
    ) -> None:
        self.config = config
        self.server = server
        self.watchers = watchers
        self.stop_event = stop_event
        self._cursor = 0

    async def run(self) -> None:
        keys = list(self.watchers)
        if self.config.server.events_polling.enabled:
            await self._events_loop(keys)
        else:
            await self._interval_loop(keys)

    async def _events_loop(self, keys: list[str]) -> None:
        wait = self.config.server.events_polling.wait_seconds
        while not self.stop_event.is_set():
            try:
                result = await self.server.get_events(keys, self._cursor, wait)
                if self.stop_event.is_set():
                    return
                if result is None:
                    await self._degraded_round(keys)
                    continue
                events, self._cursor = result
                for key, kinds in events.items():
                    watcher = self.watchers.get(key)
                    if watcher is None:
                        continue
                    for kind in kinds:
                        if kind == "restart":
                            await self._signal_restart(watcher)
                        elif kind == "incoming":
                            watcher.incoming_event.set()
                if events:
                    await self._sleep(1.0)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The listener is the only path for restart signals; it must
                # survive anything and degrade rather than die.
                logger.exception("events round failed; degrading to interval polling")
                await self._degraded_round(keys)

    async def _signal_restart(self, watcher: ChannelWatcher) -> None:
        if watcher.restart_event.is_set():
            # Already being handled: do not re-trigger, and do not ack,
            # that could wipe a freshly POSTed request (docs/02).
            return
        logger.info("[%s] restart requested by server", watcher.streamer.key)
        watcher.restart_event.set()
        await self.server.ack_restart(watcher.streamer.key)

    async def _degraded_round(self, keys: list[str]) -> None:
        for key in keys:
            watcher = self.watchers[key]
            if not watcher.restart_event.is_set() and await self.server.get_restart(key):
                await self._signal_restart(watcher)
            watcher.incoming_event.set()
        await self._sleep(self.config.server.incoming_polling.interval_seconds)

    async def _interval_loop(self, keys: list[str]) -> None:
        interval = self.config.server.incoming_polling.interval_seconds
        while not self.stop_event.is_set():
            for key in keys:
                watcher = self.watchers[key]
                if not watcher.restart_event.is_set() and await self.server.get_restart(key):
                    await self._signal_restart(watcher)
            await self._sleep(interval)

    async def _sleep(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
