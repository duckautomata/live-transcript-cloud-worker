from __future__ import annotations

import asyncio
import os
import shutil
import struct
import subprocess
import sys

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


class _HangingFfmpeg:
    """Stands in for an ffmpeg process that never finishes decoding."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False
        self.started = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        self.started.set()
        await asyncio.Event().wait()  # never returns
        raise AssertionError("unreachable")

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int | None:
        return self.returncode


@pytest.fixture
def hanging_ffmpeg(monkeypatch) -> _HangingFfmpeg:
    proc = _HangingFfmpeg()

    async def fake_exec(*args, **kwargs) -> _HangingFfmpeg:
        return proc

    monkeypatch.setattr(audio.asyncio, "create_subprocess_exec", fake_exec)
    return proc


async def test_extract_pcm_timeout_kills_ffmpeg(tmp_path, hanging_ffmpeg):
    with pytest.raises(audio.AudioExtractError, match="timed out"):
        await audio.extract_pcm("ffmpeg", tmp_path / "chunk.ts", timeout=0.05)
    assert hanging_ffmpeg.killed


async def test_extract_pcm_cancelled_kills_ffmpeg(tmp_path, hanging_ffmpeg):
    """A capture cancelled mid-decode (worker shutdown, a cancelled run) must
    not leave its ffmpeg running: nothing would ever read its output."""
    task = asyncio.create_task(audio.extract_pcm("ffmpeg", tmp_path / "chunk.ts"))
    await hanging_ffmpeg.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert hanging_ffmpeg.killed


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell script as the fake ffmpeg")
async def test_extract_pcm_cancelled_kills_a_real_process(tmp_path):
    pid_file = tmp_path / "pid"
    fake = tmp_path / "ffmpeg"
    fake.write_text(f'#!/bin/sh\necho $$ > "{pid_file}"\nexec sleep 30\n')
    fake.chmod(0o755)

    task = asyncio.create_task(audio.extract_pcm(str(fake), tmp_path / "chunk.ts"))
    for _ in range(500):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.01)
    pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Killed and reaped: the pid is gone, not a lingering sleep (or a zombie).
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
