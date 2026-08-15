"""Stream capture: yt-dlp piped into ffmpeg's segment muxer, at the live edge.

One pipeline shape for every platform (docs/03):

    yt-dlp ... -o -  |  ffmpeg -i pipe:0 -c copy -f segment
                            -segment_time N -segment_format mpegts
                            -reset_timestamps 1  chunk%06d.ts

ffmpeg cuts keyframe-aligned MPEG-TS segments; segment N is safe to consume
once chunk N+1 exists (proving N is complete) or both processes have exited.

Capture joins at the live edge (never ``--live-from-start``), so timestamps
are estimated: each segment is anchored independently at
``mtime - duration - live_latency``, ffmpeg stamps a segment's mtime when it
finishes writing it, so the anchor is a real ingestion clock that self-heals
after stalls instead of drifting. All lines are ``vodAccurate: false``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from . import audio
from .config import Config, StreamerConfig
from .models import Chunk, MediaType, StreamInfo
from .state import ChannelState
from .ytdlp import auth_args

logger = logging.getLogger(__name__)

VIDEO_FORMAT = "bestvideo[vcodec^=avc]+bestaudio[acodec^=mp4a]/best[vcodec^=avc]/best"
AUDIO_FORMAT = "bestaudio[acodec^=mp4a]/ba/best"

POLL_INTERVAL = 0.5
TERMINATE_GRACE = 5.0


@dataclass
class CaptureStats:
    segments_emitted: int = 0


ShouldStop = Callable[[], bool]


def _ytdlp_cmd(config: Config, streamer: StreamerConfig, url: str) -> list[str]:
    fmt = VIDEO_FORMAT if streamer.media_type is MediaType.VIDEO else AUDIO_FORMAT
    return [
        config.capture.yt_dlp_path,
        "--quiet",
        "--no-warnings",
        *auth_args(config, url, "download"),
        "-f",
        fmt,
        "-o",
        "-",
        url,
    ]


def _ffmpeg_cmd(config: Config, segment_dir: Path) -> list[str]:
    return [
        config.capture.ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-f",
        "segment",
        "-segment_time",
        str(config.server.buffer_size_seconds),
        "-segment_format",
        "mpegts",
        "-reset_timestamps",
        "1",
        str(segment_dir / "chunk%06d.ts"),
    ]


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=TERMINATE_GRACE)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


async def capture_stream(
    config: Config,
    streamer: StreamerConfig,
    state: ChannelState,
    info: StreamInfo,
    emit: Callable[[Chunk], Awaitable[None]],
    should_stop: ShouldStop,
) -> CaptureStats:
    """Run live-edge capture for one stream until it ends or we're told to
    stop. Every chunk goes out through ``emit`` (which may block for
    backpressure)."""
    key = streamer.key
    stats = CaptureStats()
    session_dir = state.segments_dir / f"s{int(time.time())}_{os.getpid()}"
    session_dir.mkdir(parents=True, exist_ok=True)

    ytdlp_cmd = _ytdlp_cmd(config, streamer, info.url)
    logger.info("[%s] capture start (live-edge) stream=%s: %s", key, info.stream_id, " ".join(ytdlp_cmd))
    state.ytdlp_log.append(f"--- {time.strftime('%Y-%m-%dT%H:%M:%S')} capture {info.stream_id}: {' '.join(ytdlp_cmd)}")

    # Plumb yt-dlp stdout straight into ffmpeg stdin via an OS pipe, with
    # yt-dlp stderr appended to the per-channel log for post-mortems. The
    # log opens before the pipe so an open() failure can't leak the fds.
    stderr_f = open(state.ytdlp_log.path, "ab")  # noqa: SIM115, lifetime spans the session
    read_fd, write_fd = os.pipe()
    ytdlp = None
    try:
        ytdlp = await asyncio.create_subprocess_exec(
            *ytdlp_cmd,
            stdout=write_fd,
            stderr=stderr_f,
        )
        ffmpeg = await asyncio.create_subprocess_exec(
            *_ffmpeg_cmd(config, session_dir),
            stdin=read_fd,
            stderr=stderr_f,
        )
    except OSError as exc:
        logger.error("[%s] failed to spawn capture processes: %s", key, exc)
        if ytdlp is not None:
            await _terminate(ytdlp)
        stderr_f.close()
        return stats
    finally:
        # The children hold their own copies of the pipe ends; close ours
        # exactly once (a second close could hit a recycled fd number).
        os.close(read_fd)
        os.close(write_fd)

    seq = 0
    last_progress = time.time()
    stale_after = config.server.stale_threshold.ytdlp_seconds
    stopping = False

    try:
        while True:
            if should_stop() and not stopping:
                stopping = True
                logger.info("[%s] stop requested; terminating capture", key)
                await _terminate(ytdlp)
                await _terminate(ffmpeg)

            current = session_dir / f"chunk{seq:06d}.ts"
            nxt = session_dir / f"chunk{seq + 1:06d}.ts"
            both_done = ytdlp.returncode is not None and ffmpeg.returncode is not None

            if nxt.exists() or (both_done and current.exists()):
                chunk = await _consume_segment(config, streamer, info, current)
                seq += 1
                last_progress = time.time()
                # The children write to the log via their own O_APPEND fds,
                # so the cap must be enforced from here during the session.
                state.ytdlp_log.enforce_cap()
                if chunk is not None:
                    if chunk.duration >= config.transcription.min_chunk_seconds:
                        stats.segments_emitted += 1
                        await emit(chunk)
                    else:
                        # Too short to be a real line (docs/03), no ID is
                        # ever assigned.
                        chunk.path.unlink(missing_ok=True)
                continue

            if both_done:
                logger.info(
                    "[%s] capture ended (yt-dlp rc=%s, %d segments)",
                    key,
                    ytdlp.returncode,
                    stats.segments_emitted,
                )
                break

            if not stopping and time.time() - last_progress > stale_after:
                logger.warning("[%s] no new segment for %.0fs; terminating wedged capture", key, stale_after)
                await _terminate(ytdlp)
                await _terminate(ffmpeg)
                # Loop once more to flush whatever segments now exist.
                stopping = True
                continue

            await asyncio.sleep(POLL_INTERVAL)
    finally:
        await _terminate(ytdlp)
        await _terminate(ffmpeg)
        stderr_f.close()
        # Segments at index >= seq were never consumed, delete them. Files
        # below seq belong to emitted chunks still travelling through the
        # pipeline by path; the pipeline deletes (or queues) each one, and
        # the empty session dir is swept at the next worker boot.
        consumed = {session_dir / f"chunk{i:06d}.ts" for i in range(seq)}
        with contextlib.suppress(OSError):
            for p in session_dir.iterdir():
                if p not in consumed:
                    p.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                session_dir.rmdir()

    return stats


async def _consume_segment(
    config: Config,
    streamer: StreamerConfig,
    info: StreamInfo,
    path: Path,
) -> Chunk | None:
    """Measure a finished segment and build its Chunk.

    Returns None for garbage (undecodable / zero-length audio), those emit
    no line.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    try:
        pcm = await audio.extract_pcm(config.capture.ffmpeg_path, path)
    except audio.AudioExtractError as exc:
        logger.warning("[%s] unreadable segment %s: %s", streamer.key, path.name, exc)
        path.unlink(missing_ok=True)
        return None
    duration = audio.pcm_duration(pcm)
    if duration <= 0:
        logger.debug("[%s] zero-duration segment %s; skipping", streamer.key, path.name)
        path.unlink(missing_ok=True)
        return None

    return Chunk(
        key=streamer.key,
        stream_id=info.stream_id,
        path=path,
        audio_start_time=mtime - duration - config.capture.live_latency_seconds,
        duration=duration,
        vod_accurate=False,
        media_type=streamer.media_type,
        pcm=pcm,
    )
