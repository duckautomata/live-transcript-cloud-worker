"""Audio extraction for transcription.

Each MPEG-TS chunk is decoded once to 16 kHz mono PCM with ffmpeg. That
single pass yields both:

- the exact duration (decoded sample count / sample rate, the master clock
  per docs/03; container metadata drifts), and
- the transcription payload (PCM wrapped in a WAV header, which both
  Cloudflare and Deepgram accept).

A 6-second chunk is ~192 KB of WAV, well inside every provider limit.
"""

from __future__ import annotations

import asyncio
import logging
import re
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # s16le mono


class AudioExtractError(Exception):
    pass


async def extract_pcm(ffmpeg_path: str, media_path: Path, timeout: float = 30.0) -> bytes:
    """Decode a media file's audio track to raw s16le 16 kHz mono PCM."""
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            pcm, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise AudioExtractError(f"ffmpeg timed out decoding {media_path.name}") from None
    except OSError as exc:
        raise AudioExtractError(f"could not run ffmpeg: {exc}") from exc

    if proc.returncode != 0 and not pcm:
        raise AudioExtractError(f"ffmpeg failed on {media_path.name} (rc={proc.returncode}): {stderr.decode(errors='replace')[-300:]}")
    return pcm


def pcm_duration(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_SECOND


_DURATION_RE = re.compile(rb"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")


async def container_duration(ffmpeg_path: str, media_path: Path, timeout: float = 15.0) -> float:
    """Container-metadata duration, from ffmpeg's probe banner.

    The docs/03 fallback clock: used only when a span has no decodable audio
    (e.g. a video-only DASH sequence), decoded sample count stays the master
    clock everywhere else. Returns 0.0 when no duration is discernible.
    """
    cmd = [ffmpeg_path, "-hide_banner", "-i", str(media_path)]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return 0.0
    except OSError:
        return 0.0
    match = _DURATION_RE.search(stderr)
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw s16le 16 kHz mono PCM in a minimal RIFF/WAVE header."""
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # mono
        SAMPLE_RATE,
        BYTES_PER_SECOND,
        2,  # block align
        16,  # bits per sample
        b"data",
        len(pcm),
    )
    return header + pcm
