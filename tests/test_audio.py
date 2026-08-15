from __future__ import annotations

import shutil
import struct
import subprocess

import pytest

from live_transcript_cloud_worker import audio

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")


def test_wav_header_shape():
    pcm = b"\x00\x01" * 16000  # 1 second
    wav = audio.pcm_to_wav(pcm)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) == 44 + len(pcm)
    sample_rate = struct.unpack_from("<I", wav, 24)[0]
    assert sample_rate == 16000


def test_pcm_duration():
    assert audio.pcm_duration(b"\x00" * audio.BYTES_PER_SECOND * 3) == pytest.approx(3.0)
    assert audio.pcm_duration(b"") == 0.0


async def test_extract_pcm_roundtrip(tmp_path):
    src = tmp_path / "tone.ts"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "-f", "mpegts", str(src)],
        check=True,
    )
    pcm = await audio.extract_pcm(ffmpeg, src)
    duration = audio.pcm_duration(pcm)
    assert 1.8 <= duration <= 2.3  # AAC padding makes it slightly inexact
    wav = audio.pcm_to_wav(pcm)
    assert wav[:4] == b"RIFF"


async def test_extract_pcm_garbage_raises_or_empty(tmp_path):
    src = tmp_path / "garbage.ts"
    src.write_bytes(b"\x00" * 1024)
    try:
        pcm = await audio.extract_pcm(ffmpeg, src)
    except audio.AudioExtractError:
        return
    assert audio.pcm_duration(pcm) == 0.0
