"""YouTube DASH live-from-start capture (docs/03 strategy 3).

``yt-dlp --live-from-start --keep-fragments`` writes numbered fragment files;
this module groups them by sequence, merges each sequence's tracks into one
MPEG-TS with ffmpeg, accumulates merged sequences until
``buffer_size_seconds - 0.2`` of measured audio, and emits that as one chunk.
Because capture starts at second zero, timestamps are exact
(``vodAccurate: true``): ``audio_start_time = stream_start + accumulated
durations``.

All the machinery here exists because live DASH is fragile (docs/03):

- **Readiness**: a video-mode sequence needs 2 files (video+audio) or 1 file
  carrying both tracks; audio mode needs 1. Incomplete sequences are forced
  through with partial data after ``stale_threshold.fragment_seconds``.
- **Stall watchdog**: no new fragment for ``stale_threshold.ytdlp_seconds``
  terminates yt-dlp and finishes cleanly, no live-edge switch, because with
  no fragments there is no live edge to catch up to and switching would
  discard the buffer.
- **Resume state**: ``{stream_id, last_sequence, current_stream_time}`` is
  written atomically after each emitted chunk so a worker restart continues
  instead of replaying the stream as duplicate lines.
- **Continuity verification on resume**: the previously downloaded Frag1 is
  backed up and byte-compared against the freshly downloaded one. Identical
  → same stream, resume. Different → the platform reset the sequence
  numbering: reset ``last_sequence`` to 0 but KEEP ``current_stream_time``
  (the earlier content is still in the VOD), wipe the fragment dir, restart
  yt-dlp. Timeout waiting for the new Frag1 → the stream is far ahead;
  restore the backup and resume without a reset.
- **Gap check**: after each emitted chunk, being more than
  ``stale_threshold.lfs_gap_seconds`` behind live abandons catch-up and the
  caller switches to live-edge capture.

Must not be used with cookies enabled (enforced at config validation): a
--live-from-start backfill hammers the platform with a long-lived
authenticated session, exactly the fingerprint that gets accounts flagged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from . import audio
from .capture import AUDIO_FORMAT, VIDEO_FORMAT, ShouldStop, _terminate
from .config import Config, StreamerConfig
from .models import Chunk, MediaType, StreamInfo
from .state import ChannelState, atomic_write
from .ytdlp import auth_args

logger = logging.getLogger(__name__)

_FRAG_RE = re.compile(r"Frag(\d+)")
VERIFICATION_WAIT_SECONDS = 90.0
MERGE_TIMEOUT_SECONDS = 30.0
BUFFER_SLACK_SECONDS = 0.2  # fragment durations aren't exact; 5.99s ≈ 6s
POLL_INTERVAL = 1.0


@dataclass
class DashStats:
    segments_emitted: int = 0
    # True when the caller should continue this stream with live-edge capture
    # (fell too far behind live, or yt-dlp failed before producing anything).
    fell_behind: bool = False


# ------------------------------------------------------------- resume state


def resume_point(state: ChannelState, stream_id: str, default: float) -> float:
    """The stream time capture would resume from, used by the pre-flight
    gap check before DASH is even attempted."""
    seq, stream_time = _load_state(_state_path(state), stream_id, default)
    return stream_time


def _state_path(state: ChannelState) -> Path:
    return state.root / "dash_state.json"


def _load_state(path: Path, stream_id: str, default_time: float) -> tuple[int, float]:
    try:
        data = json.loads(path.read_text())
        if data.get("stream_id") == stream_id:
            return int(data.get("last_sequence", 0)), float(data.get("current_stream_time", default_time))
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("dash state unreadable (%s); starting fresh", exc)
    return 0, default_time


def _save_state(path: Path, stream_id: str, last_seq: int, stream_time: float) -> None:
    try:
        atomic_write(
            path,
            json.dumps(
                {
                    "stream_id": stream_id,
                    "last_sequence": last_seq,
                    "current_stream_time": stream_time,
                }
            ).encode(),
        )
    except OSError as exc:
        logger.warning("dash state save failed: %s", exc)


# ---------------------------------------------------------------- processes


def _ytdlp_cmd(config: Config, streamer: StreamerConfig, url: str, fragment_dir: Path) -> list[str]:
    video = streamer.media_type is MediaType.VIDEO
    return [
        config.capture.yt_dlp_path,
        "--live-from-start",
        "--keep-fragments",
        "--no-progress",
        "--no-colors",
        *auth_args(config, url, "download"),
        # General HTTP connection retries.
        "--retries",
        "20",
        # Individual stream chunks (transient CDN issues).
        "--fragment-retries",
        "10",
        # YouTube parsing errors (e.g. "Video is no longer live").
        "--extractor-retries",
        "5",
        # Kill hanging connections and force a retry.
        "--socket-timeout",
        "30",
        "--retry-sleep",
        "fragment:exp=1:10",
        "--retry-sleep",
        "http:exp=1:60",
        "--retry-sleep",
        "extractor:exp=1:60",
        "--hls-prefer-native",
        "--hls-use-mpegts",
        "-f",
        VIDEO_FORMAT if video else AUDIO_FORMAT,
        "-o",
        f"{fragment_dir}/%(id)s.%(format_id)s",
        url,
    ]


async def _spawn_ytdlp(
    config: Config,
    streamer: StreamerConfig,
    state: ChannelState,
    info: StreamInfo,
    fragment_dir: Path,
) -> asyncio.subprocess.Process | None:
    # Delete any completed final output first, or yt-dlp says "already
    # downloaded" and exits without producing fragments.
    with contextlib.suppress(OSError):
        for path in fragment_dir.glob(f"{info.stream_id}*"):
            if path.is_file() and "Frag" not in path.name:
                path.unlink(missing_ok=True)
                logger.info(
                    "[%s] deleted final file %s so yt-dlp doesn't skip the download",
                    streamer.key,
                    path.name,
                )

    cmd = _ytdlp_cmd(config, streamer, info.url, fragment_dir)
    state.ytdlp_log.append(f"--- {time.strftime('%Y-%m-%dT%H:%M:%S')} dash capture {info.stream_id}: {' '.join(cmd)}")
    stderr_f = open(state.ytdlp_log.path, "ab")  # noqa: SIM115, child keeps its own fd
    try:
        return await asyncio.create_subprocess_exec(*cmd, stdout=stderr_f, stderr=stderr_f)
    except OSError as exc:
        logger.error("[%s] failed to spawn yt-dlp for DASH: %s", streamer.key, exc)
        return None
    finally:
        stderr_f.close()


async def _merge_fragments(config: Config, key: str, state: ChannelState, inputs: list[Path], output: Path) -> bool:
    """Merge one sequence's fragment files into a single MPEG-TS."""
    cmd = [config.capture.ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error"]
    for path in inputs:
        cmd += ["-i", str(path)]
    cmd += ["-c", "copy", "-f", "mpegts", str(output)]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=MERGE_TIMEOUT_SECONDS)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("[%s] ffmpeg merge timed out for %s", key, output.name)
            return False
    except OSError as exc:
        logger.error("[%s] could not run ffmpeg merge: %s", key, exc)
        return False
    if proc.returncode != 0:
        state.ytdlp_log.append(f"--- ffmpeg merge failed: {' '.join(cmd)}\n{stderr.decode(errors='replace')[-2000:]}")
        logger.warning("[%s] ffmpeg merge failed for %s", key, output.name)
        return False
    return True


def _delete_consumed_fragments(files: list[Path], seq: int) -> None:
    """Delete a processed sequence's fragment files.

    Deliberate deviation from the reference worker, which keeps every
    fragment until the next stream's wipe: a 6-hour 1080p stream leaves
    15-20 GB on disk otherwise. Sequence 1 is always kept, its Frag1 is
    what the continuity verification byte-compares on resume.
    """
    if seq == 1:
        return
    for path in files:
        path.unlink(missing_ok=True)


# ------------------------------------------------------------ fragment scan


def _scan_fragments(fragment_dir: Path, last_seq: int) -> dict[int, list[Path]]:
    """Group valid fragment files by sequence number, newest state on disk."""
    pending: dict[int, list[Path]] = {}
    try:
        entries = list(fragment_dir.iterdir())
    except OSError:
        return pending
    for path in entries:
        name = path.name
        if "Frag" not in name or name.endswith((".part", ".ytdl", ".bak")):
            continue
        match = _FRAG_RE.search(name)
        if not match:
            continue
        seq = int(match.group(1))
        if seq <= last_seq:
            continue
        try:
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue
        pending.setdefault(seq, []).append(path)
    return pending


# ------------------------------------------------------------- verification


def _is_content_identical(file1: Path, file2: Path, buffer_size: int = 65536) -> bool:
    """Strict byte-to-byte comparison (size short-circuit first)."""
    if file1.stat().st_size != file2.stat().st_size:
        return False
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            b1 = f1.read(buffer_size)
            b2 = f2.read(buffer_size)
            if b1 != b2:
                return False
            if not b1:
                return True


def _setup_verification(key: str, fragment_dir: Path) -> tuple[Path, str] | None:
    """Back up the previously downloaded Frag1 for the continuity check.

    Picks the lowest format id lexicographically (f140 sorts before f299),
    matching the reference worker.
    """
    candidates = sorted(
        p for p in fragment_dir.glob("*Frag1") if p.name.endswith("Frag1") and not p.name.endswith((".part", ".ytdl", ".bak"))
    )
    if not candidates:
        logger.warning("[%s] no Frag1 found for verification; skipping continuity check", key)
        return None
    selected = candidates[0]
    backup = selected.with_name(selected.name + ".bak")
    try:
        shutil.move(selected, backup)
    except OSError as exc:
        logger.warning("[%s] verification setup failed: %s", key, exc)
        return None
    logger.info("[%s] selected %s for continuity verification", key, selected.name)
    return backup, selected.name


async def _verify_continuity(
    config: Config,
    streamer: StreamerConfig,
    state: ChannelState,
    info: StreamInfo,
    fragment_dir: Path,
    process: asyncio.subprocess.Process,
    backup: Path,
    target_name: str,
    last_seq: int,
    stream_time: float,
    should_stop: ShouldStop,
) -> tuple[int, float, asyncio.subprocess.Process | None]:
    """Byte-compare the fresh Frag1 against the backup; on mismatch the
    platform reset its numbering: keep the stream clock, reset the sequence,
    wipe the dir, restart yt-dlp."""
    key = streamer.key
    logger.info("[%s] verifying stream continuity via Frag1...", key)
    target = fragment_dir / target_name
    deadline = time.monotonic() + VERIFICATION_WAIT_SECONDS
    fresh = None
    while time.monotonic() < deadline:
        # File check first: a fast-exiting yt-dlp may have written the
        # fragment on its way out.
        if target.exists():
            fresh = target
            break
        if should_stop() or process.returncode is not None:
            break
        await asyncio.sleep(1)

    if fresh is None:
        # Timeout: the stream is far ahead and yt-dlp isn't refetching Frag1.
        logger.warning(
            "[%s] verification timeout: new Frag1 never appeared; assuming same stream (no reset)",
            key,
        )
        with contextlib.suppress(OSError):
            shutil.move(backup, target)
        return last_seq, stream_time, process

    try:
        identical = await asyncio.to_thread(_is_content_identical, fresh, backup)
    except OSError as exc:
        logger.warning("[%s] verification compare failed (%s); assuming same stream", key, exc)
        with contextlib.suppress(OSError):
            shutil.move(backup, fresh)
        return last_seq, stream_time, process

    if identical:
        logger.info("[%s] continuity verified: same stream, resuming from seq %d", key, last_seq)
        backup.unlink(missing_ok=True)
        return last_seq, stream_time, process

    logger.warning("[%s] sequence reset detected! Restarting from Frag1, keeping stream clock", key)
    await _terminate(process)
    await asyncio.to_thread(shutil.rmtree, fragment_dir, ignore_errors=True)
    fragment_dir.mkdir(parents=True, exist_ok=True)
    _save_state(_state_path(state), info.stream_id, 0, stream_time)
    new_process = await _spawn_ytdlp(config, streamer, state, info, fragment_dir)
    return 0, stream_time, new_process


# ------------------------------------------------------------- capture loop


async def capture_stream_dash(
    config: Config,
    streamer: StreamerConfig,
    state: ChannelState,
    info: StreamInfo,
    emit: Callable[[Chunk], Awaitable[None]],
    should_stop: ShouldStop,
) -> DashStats:
    key = streamer.key
    stats = DashStats()
    fragment_dir = state.root / "fragments"
    state_path = _state_path(state)
    session_dir = state.segments_dir / f"d{int(time.time())}_{os.getpid()}"
    session_dir.mkdir(parents=True, exist_ok=True)

    initial_start = float(info.start_time or time.time())
    last_seq, stream_time = _load_state(state_path, info.stream_id, initial_start)

    verification: tuple[Path, str] | None = None
    if last_seq == 0 and stream_time == initial_start:
        logger.info("[%s] DASH: new stream (or no state); cleaning fragment dir", key)
        await asyncio.to_thread(shutil.rmtree, fragment_dir, ignore_errors=True)
        fragment_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("[%s] DASH: resuming from seq %d at stream time %.0f", key, last_seq, stream_time)
        fragment_dir.mkdir(parents=True, exist_ok=True)
        verification = _setup_verification(key, fragment_dir)

    logger.info("[%s] capture start (DASH) stream=%s", key, info.stream_id)
    process = await _spawn_ytdlp(config, streamer, state, info, fragment_dir)
    if process is None:
        stats.fell_behind = True  # let the caller try live-edge instead
        return stats

    start_seq = last_seq
    is_video = streamer.media_type is MediaType.VIDEO
    thresholds = config.server.stale_threshold

    # The chunk under construction: merged sequences appended on disk, their
    # decoded PCM accumulated alongside so nothing is decoded twice.
    chunk_index = 0
    chunk_path = session_dir / f"chunk{chunk_index:06d}.ts"
    chunk_pcm = bytearray()
    buffer_duration = 0.0

    current_processing_seq: int | None = None
    seq_first_seen = 0.0
    last_new_fragment = time.time()
    watchdog_fired = False

    async def emit_buffer() -> None:
        nonlocal chunk_index, chunk_path, chunk_pcm, buffer_duration, stream_time
        chunk = Chunk(
            key=key,
            stream_id=info.stream_id,
            path=chunk_path,
            audio_start_time=stream_time,
            duration=buffer_duration,
            vod_accurate=True,
            media_type=streamer.media_type,
            pcm=bytes(chunk_pcm),
        )
        stats.segments_emitted += 1
        await emit(chunk)
        stream_time += buffer_duration
        _save_state(state_path, info.stream_id, last_seq, stream_time)
        chunk_index += 1
        chunk_path = session_dir / f"chunk{chunk_index:06d}.ts"
        chunk_pcm = bytearray()
        buffer_duration = 0.0

    clean_end = False
    try:
        # Everything below, verification included, runs under the finally
        # that owns the yt-dlp process: no unwind path may orphan a
        # --live-from-start download.
        if verification is not None:
            last_seq, stream_time, process = await _verify_continuity(
                config,
                streamer,
                state,
                info,
                fragment_dir,
                process,
                verification[0],
                verification[1],
                last_seq,
                stream_time,
                should_stop,
            )
            if process is None:
                logger.error("[%s] DASH process could not be restarted; falling back", key)
                stats.fell_behind = True
                return stats
            start_seq = last_seq
            # A process that already exited is handled by the monitor loop:
            # any fragments it left behind are drained before classification.

        while not should_stop():
            pending = _scan_fragments(fragment_dir, last_seq)
            if not pending:
                # Only classify the exit once every fragment left on disk
                # has been drained, fragments written just before yt-dlp
                # exited must not be lost.
                if process.returncode is not None:
                    if watchdog_fired:
                        # A stalled stream has no live edge to catch up to;
                        # finish cleanly rather than switching (docs/03).
                        logger.info("[%s] DASH finished after stall watchdog", key)
                    elif process.returncode != 0 and last_seq == start_seq:
                        logger.warning(
                            "[%s] yt-dlp exited rc=%s before producing any fragments; switching to live-edge now",
                            key,
                            process.returncode,
                        )
                        stats.fell_behind = True
                    elif process.returncode != 0:
                        logger.warning("[%s] yt-dlp exited rc=%s after seq %d", key, process.returncode, last_seq)
                    else:
                        logger.info("[%s] DASH yt-dlp ended", key)
                        clean_end = True
                    break
                # Stall watchdog: yt-dlp alive but producing nothing. Finish
                # cleanly; do NOT switch to live-edge (nothing to catch up to,
                # and switching would discard the buffer).
                if time.time() - last_new_fragment > thresholds.ytdlp_seconds:
                    logger.warning(
                        "[%s] no new DASH fragments in %.0fs; terminating wedged yt-dlp",
                        key,
                        time.time() - last_new_fragment,
                    )
                    watchdog_fired = True
                    await _terminate(process)
                    continue  # one more scan drains anything just written
                await asyncio.sleep(POLL_INTERVAL)
                continue

            last_new_fragment = time.time()
            state.ytdlp_log.enforce_cap()

            for seq in sorted(pending):
                if should_stop():
                    break
                files = sorted(pending[seq])

                if current_processing_seq != seq:
                    current_processing_seq = seq
                    seq_first_seen = time.time()

                # Readiness: video mode wants both tracks; audio mode one file.
                is_ready = len(files) >= 2 if is_video else len(files) >= 1
                if not is_ready and time.time() - seq_first_seen > thresholds.fragment_seconds:
                    logger.warning(
                        "[%s] seq %d incomplete (%d file(s)) but stale; processing partial data",
                        key,
                        seq,
                        len(files),
                    )
                    is_ready = True
                if not is_ready:
                    # Still downloading; wait for the next scan.
                    break

                merged = fragment_dir / f"merged_{seq}.ts"
                if not await _merge_fragments(config, key, state, files, merged):
                    merged.unlink(missing_ok=True)  # ffmpeg -y creates it before failing
                    if time.time() - seq_first_seen <= thresholds.fragment_seconds:
                        # Transient (host load, merge timeout): retry on the
                        # next scan rather than silently dropping a sequence
                        # the VOD contains.
                        break
                    # Persistent failure: give the sequence up, but keep the
                    # vodAccurate clock aligned with an estimate of its span.
                    estimate = await audio.container_duration(config.capture.ffmpeg_path, files[0])
                    logger.error(
                        "[%s] merge kept failing for seq %d; skipping it (clock advanced %.2fs)",
                        key,
                        seq,
                        estimate,
                    )
                    buffer_duration += estimate
                    _delete_consumed_fragments(files, seq)
                    last_seq = seq
                    continue

                try:
                    pcm: bytes = await audio.extract_pcm(config.capture.ffmpeg_path, merged)
                    duration = audio.pcm_duration(pcm)
                except audio.AudioExtractError:
                    pcm = b""
                    duration = 0.0
                if duration <= 0:
                    # No decodable audio, e.g. a video-only sequence forced
                    # through with its audio fragment missing. docs/03: fall
                    # back to container duration so the vodAccurate clock
                    # still advances; otherwise every later timestamp drifts
                    # early while still claiming to be exact.
                    duration = await audio.container_duration(config.capture.ffmpeg_path, merged)
                    if duration > 0:
                        logger.warning(
                            "[%s] seq %d has no decodable audio; clock advanced %.2fs via container duration",
                            key,
                            seq,
                            duration,
                        )
                if duration > 0:
                    try:
                        with open(chunk_path, "ab") as chunk_f:
                            chunk_f.write(merged.read_bytes())
                        chunk_pcm.extend(pcm)
                    except OSError as exc:
                        logger.warning("[%s] could not buffer merged seq %d: %s", key, seq, exc)
                    buffer_duration += duration
                merged.unlink(missing_ok=True)
                _delete_consumed_fragments(files, seq)
                last_seq = seq

                if buffer_duration >= config.server.buffer_size_seconds - BUFFER_SLACK_SECONDS:
                    await emit_buffer()
                    gap = time.time() - stream_time
                    if gap > thresholds.lfs_gap_seconds:
                        logger.warning(
                            "[%s] DASH is %.0f minutes behind live; switching to live-edge",
                            key,
                            gap / 60,
                        )
                        stats.fell_behind = True
                        return stats

            await asyncio.sleep(POLL_INTERVAL)
    finally:
        if process is not None:
            await _terminate(process)
        # Flush whatever partial buffer exists; the stream clock must advance
        # even if the tail is too short to become a line.
        if buffer_duration > 0:
            if buffer_duration >= config.transcription.min_chunk_seconds:
                await emit_buffer()
            else:
                stream_time += buffer_duration
                _save_state(state_path, info.stream_id, last_seq, stream_time)
                chunk_path.unlink(missing_ok=True)
        else:
            chunk_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            session_dir.rmdir()  # only succeeds once the pipeline consumed everything

    if clean_end:
        # The stream is over; nothing left to resume. Drop the fragment dir
        # rather than letting a stream's worth of fragments sit on disk.
        await asyncio.to_thread(shutil.rmtree, fragment_dir, ignore_errors=True)

    return stats
