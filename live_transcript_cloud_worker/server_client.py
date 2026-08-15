"""HTTP client for live-transcript-server, implementing docs/01 + docs/06.

One httpx.AsyncClient (one connection pool) serves every endpoint; the
/events long-poll gets its own client so no retry logic and no shared
timeout can ever stack onto a request deliberately held open for ~25 s.

Retry classification (docs/06, "Retry policy summary"):
    network errors / timeouts   retry, exponential backoff + jitter
    5xx                         retry, bounded
    429                         retry, honouring Retry-After
    400 / 403 / 404             permanent (except 404 on /media)
    409 on /line                not a retry, the caller must /sync
    200 / 204 / 208             success
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from .config import Config
from .models import Line, MediaType

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
SYNC_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
MEDIA_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)

BACKOFF_BASE = 1.0
BACKOFF_CAP = 30.0


class LineResult(Enum):
    OK = "ok"
    CONFLICT = "conflict"  # 409: caller must POST /sync with full state
    FAILED = "failed"  # exhausted retries or permanent error; do not resend


class MediaResult(Enum):
    OK = "ok"
    DROP = "drop"  # permanent failure or server kept rejecting; delete the file
    # The server was unreachable (or rate-limiting); keep the file in the
    # durable queue and try again later, docs/06: a server outage must
    # never drop media.
    RETRY_LATER = "retry_later"


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), BACKOFF_CAP)
        except ValueError:
            pass
    return min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt)) * (0.5 + random.random())


class ServerClient:
    """Talks to the real server. See LocalClient for server.enabled: false."""

    def __init__(self, config: Config, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        base = config.server.url
        headers = {"X-API-Key": config.server.api_key}
        self._client = httpx.AsyncClient(base_url=base, headers=headers, timeout=DEFAULT_TIMEOUT, transport=transport)
        self._longpoll = httpx.AsyncClient(base_url=base, headers=headers, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._longpoll.aclose()

    # ------------------------------------------------------------ plumbing

    async def _request(
        self,
        method: str,
        path: str,
        *,
        attempts: int = 4,
        success: tuple[int, ...] = (200, 204, 208),
        passthrough: tuple[int, ...] = (),
        retry_404: bool = False,
        **kwargs: Any,
    ) -> httpx.Response | None:
        """Issue a request with the docs/06 retry classification.

        Returns the response when its status is in ``success`` or
        ``passthrough`` (caller handles it), or None when the request
        permanently failed / exhausted its retries.
        """
        last_error: str = ""
        for attempt in range(attempts):
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(_backoff(attempt))
                continue

            status = response.status_code
            if status in success or status in passthrough:
                return response
            if status == 403:
                logger.error("%s %s -> 403; check server.apiKey", method, path)
                return None
            if status == 429 or status >= 500 or (status == 404 and retry_404):
                last_error = f"HTTP {status}"
                await asyncio.sleep(_backoff(attempt, response.headers.get("Retry-After")))
                continue
            body = response.text[:200]
            logger.error("%s %s -> HTTP %s (%s); not retrying", method, path, status, body)
            return None

        logger.warning("%s %s failed after %d attempts (%s)", method, path, attempts, last_error)
        return None

    # ------------------------------------------------------------ lifecycle

    async def activate(self, channel: str, stream_id: str, title: str, start_time: int, media_type: MediaType) -> bool:
        params = {
            "id": stream_id,
            "title": title or "untitled",
            "startTime": str(start_time),
            "mediaType": media_type.value,
        }
        response = await self._request("POST", f"/{channel}/activate", params=params, attempts=5)
        return response is not None

    async def deactivate(self, channel: str, stream_id: str) -> bool:
        if not stream_id:
            return True  # nothing to deactivate; empty id would 400
        response = await self._request("POST", f"/{channel}/deactivate", params={"id": stream_id}, attempts=3)
        return response is not None

    # ------------------------------------------------------------ hot path

    async def post_line(self, channel: str, stream_id: str, line: Line) -> LineResult:
        response = await self._request(
            "POST",
            f"/{channel}/line/{stream_id}",
            json=line.to_wire(),
            attempts=5,
            passthrough=(409,),
        )
        if response is None:
            return LineResult.FAILED
        if response.status_code == 409:
            return LineResult.CONFLICT
        return LineResult.OK

    async def sync(self, channel: str, payload: dict[str, Any]) -> bool:
        response = await self._request("POST", f"/{channel}/sync", json=payload, attempts=4, timeout=SYNC_TIMEOUT)
        return response is not None

    async def upload_media(self, channel: str, stream_id: str, line_id: int, path: Path) -> MediaResult:
        """Stream a chunk file into the multipart body.

        Retry semantics differ by failure class (docs/04, docs/06):
        - 404 (activate in flight) and 500 (ffmpeg/storage/DB) get bounded
          in-call retries, then the chunk is DROPped, the server saw the
          request and kept rejecting it.
        - Network errors and 429 mean the server never processed the chunk;
          exhausting the in-call budget returns RETRY_LATER so the uploader
          keeps the file and retries after a delay. A server outage must
          never destroy queued media.
        """
        reject_kind = ""  # non-empty when the *server* rejected the chunk
        for attempt in range(4):
            try:
                with open(path, "rb") as f:
                    response = await self._client.post(
                        f"/{channel}/media/{stream_id}/{line_id}",
                        files={"file": (path.name, f, "video/mp2t")},
                        timeout=MEDIA_TIMEOUT,
                    )
            except OSError as exc:
                logger.error("media %s/%d: cannot read %s: %s", stream_id, line_id, path, exc)
                return MediaResult.DROP
            except httpx.HTTPError as exc:
                reject_kind = ""
                logger.warning("media %s/%d attempt %d: %s", stream_id, line_id, attempt, exc)
                await asyncio.sleep(_backoff(attempt))
                continue

            status = response.status_code
            if status == 200:
                return MediaResult.OK
            if status == 404 or status >= 500:
                # 404 = the stream row isn't there yet (activate in flight);
                # retryable here and only here (docs/06).
                reject_kind = f"HTTP {status}"
                await asyncio.sleep(_backoff(attempt, response.headers.get("Retry-After")))
                continue
            if status == 429:
                reject_kind = ""
                await asyncio.sleep(_backoff(attempt, response.headers.get("Retry-After")))
                continue
            logger.error(
                "media %s/%d -> HTTP %s (%s); dropping",
                stream_id,
                line_id,
                status,
                response.text[:200],
            )
            return MediaResult.DROP

        if reject_kind:
            logger.warning("media %s/%d: server kept rejecting (%s); dropping", stream_id, line_id, reject_kind)
            return MediaResult.DROP
        logger.warning("media %s/%d: server unreachable; keeping file for a later attempt", stream_id, line_id)
        return MediaResult.RETRY_LATER

    # ------------------------------------------------------------ signals

    async def get_events(self, channels: list[str], since: int, wait: int) -> tuple[dict[str, list[str]], int] | None:
        """One long-poll round. Returns (events, cursor) or None on failure,
        the caller degrades to interval polling for that round (docs/01)."""
        timeout = httpx.Timeout(connect=5.0, read=wait + 10.0, write=10.0, pool=5.0)
        try:
            response = await self._longpoll.get(
                "/events",
                params={"channels": ",".join(channels), "since": since, "wait": wait},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            logger.debug("/events failed: %s", exc)
            return None
        if response.status_code == 204:
            return {}, since
        if response.status_code == 200:
            try:
                body = response.json()
                return dict(body.get("events") or {}), int(body.get("cursor", since))
            except (ValueError, TypeError, AttributeError, KeyError):
                logger.warning("/events returned unparseable body")
                return None
        if response.status_code == 403:
            logger.error("/events -> 403; check server.apiKey")
        else:
            logger.debug("/events -> HTTP %s", response.status_code)
        return None

    async def get_incoming(self, channel: str) -> list[str]:
        response = await self._request("GET", f"/{channel}/incoming", attempts=2)
        if response is None:
            return []
        try:
            return list(response.json().get("urls") or [])
        except (ValueError, TypeError):
            return []

    async def delete_incoming(self, channel: str, url: str) -> bool:
        response = await self._request(
            "DELETE",
            f"/{channel}/incoming",
            params={"url": url},
            attempts=2,
            success=(204,),
            passthrough=(404,),
        )
        return response is not None  # 404 = already gone = success

    async def get_restart(self, channel: str) -> bool:
        response = await self._request("GET", f"/{channel}/restart", attempts=1)
        if response is None:
            return False
        try:
            return bool(response.json().get("pending", False))
        except (ValueError, TypeError):
            return False

    async def ack_restart(self, channel: str) -> bool:
        response = await self._request(
            "DELETE",
            f"/{channel}/restart",
            attempts=3,
            success=(200, 204),
            passthrough=(404,),
        )
        return response is not None  # 404 = nothing pending = success

    async def post_status(self, version: str, build_time: str, keys: list[str]) -> None:
        body = {"version": version, "build_time": build_time, "keys": keys}
        try:
            response = await self._client.post("/status", json=body)
            if response.status_code != 200:
                logger.warning("/status -> HTTP %s", response.status_code)
        except httpx.HTTPError as exc:
            logger.warning("/status failed: %s", exc)

    async def server_version(self) -> dict[str, str] | None:
        try:
            response = await self._client.get("/version")
            if response.status_code == 200:
                return response.json()
        except (httpx.HTTPError, ValueError):
            pass
        return None


class LocalClient:
    """server.enabled: false, no HTTP at all; transcripts go to
    tmp/{key}/transcript.text for local development (docs/05)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._starts: dict[str, int] = {}

    def _path(self, channel: str) -> Path:
        return self.config.channel_dir(channel) / "transcript.text"

    async def aclose(self) -> None:
        pass

    async def activate(self, channel: str, stream_id: str, title: str, start_time: int, media_type: MediaType) -> bool:
        self._starts[channel] = start_time
        path = self._path(channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {title} ({stream_id}) start={start_time} ===\n")
        return True

    async def deactivate(self, channel: str, stream_id: str) -> bool:
        return True

    async def post_line(self, channel: str, stream_id: str, line: Line) -> LineResult:
        start = self._starts.get(channel, line.timestamp)
        offset = max(0, line.timestamp - start)
        stamp = time.strftime("%H:%M:%S", time.gmtime(offset))
        text = " ".join(s.text for s in line.segments).strip()
        with open(self._path(channel), "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {text}\n")
        return LineResult.OK

    async def sync(self, channel: str, payload: dict[str, Any]) -> bool:
        return True

    async def upload_media(self, channel: str, stream_id: str, line_id: int, path: Path) -> MediaResult:
        return MediaResult.OK

    async def get_events(self, channels, since, wait):
        return None

    async def get_incoming(self, channel: str) -> list[str]:
        return []

    async def delete_incoming(self, channel: str, url: str) -> bool:
        return True

    async def get_restart(self, channel: str) -> bool:
        return False

    async def ack_restart(self, channel: str) -> bool:
        return True

    async def post_status(self, version: str, build_time: str, keys: list[str]) -> None:
        pass

    async def server_version(self) -> dict[str, str] | None:
        return None
