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
import contextlib
import logging
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
            raise AudioExtractError(f"ffmpeg timed out decoding {media_path.name}") from None
        finally:
            # Timed out, or the capture calling this was cancelled: ffmpeg must
            # not outlive the call (nobody would read its output any more).
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
    except OSError as exc:
        raise AudioExtractError(f"could not run ffmpeg: {exc}") from exc

    if proc.returncode != 0 and not pcm:
        raise AudioExtractError(f"ffmpeg failed on {media_path.name} (rc={proc.returncode}): {stderr.decode(errors='replace')[-300:]}")
    return pcm


def pcm_duration(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_SECOND


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
