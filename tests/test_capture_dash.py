"""DASH capture tests: fragment machinery + end-to-end runs driven by a
scripted fake yt-dlp that drops fragment files into the watch dir."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from conftest import make_config

from live_transcript_cloud_worker.capture_dash import (
    _is_content_identical,
    _load_state,
    _save_state,
    _scan_fragments,
    _setup_verification,
    capture_stream_dash,
    resume_point,
)
from live_transcript_cloud_worker.config import (
    CaptureConfig,
    ServerConfig,
    StaleThresholdConfig,
    StreamerConfig,
)
from live_transcript_cloud_worker.models import MediaType, StreamInfo
from live_transcript_cloud_worker.state import ChannelState

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not available")

START = 1_700_000_000
INFO = StreamInfo(
    url="https://www.youtube.com/watch?v=vid1",
    stream_id="vid1",
    title="T",
    is_live=True,
    start_time=START,
)
STREAMER = StreamerConfig(key="chan", urls=("u",), media_type=MediaType.AUDIO)


# ------------------------------------------------------------- unit pieces


def test_state_roundtrip(tmp_path):
    path = tmp_path / "dash_state.json"
    _save_state(path, "vid1", 7, 123.5)
    assert _load_state(path, "vid1", 0.0) == (7, 123.5)
    # Different stream: state ignored.
    assert _load_state(path, "other", 42.0) == (0, 42.0)
    # Corrupt file: default.
    path.write_bytes(b"{nope")
    assert _load_state(path, "vid1", 42.0) == (0, 42.0)


def test_resume_point_reads_state(tmp_path):
    state = ChannelState("chan", tmp_path / "chan")
    assert resume_point(state, "vid1", 99.0) == 99.0
    _save_state(state.root / "dash_state.json", "vid1", 3, 500.0)
    assert resume_point(state, "vid1", 99.0) == 500.0


def test_scan_fragments_groups_and_filters(tmp_path):
    d = tmp_path / "frags"
    d.mkdir()
    (d / "vid1.f140.Frag1").write_bytes(b"a")
    (d / "vid1.f299.Frag1").write_bytes(b"v")
    (d / "vid1.f140.Frag2").write_bytes(b"b")
    (d / "vid1.f140.Frag3").write_bytes(b"")  # empty: skip
    (d / "vid1.f140.Frag4.part").write_bytes(b"x")  # partial: skip
    (d / "vid1.f140.Frag1.bak").write_bytes(b"x")  # backup: skip
    (d / "vid1.f140.ytdl").write_bytes(b"x")  # bookkeeping: skip

    pending = _scan_fragments(d, last_seq=0)
    assert sorted(pending) == [1, 2]
    assert len(pending[1]) == 2  # audio + video track files
    # Already-processed sequences are skipped.
    assert sorted(_scan_fragments(d, last_seq=1)) == [2]


def test_content_identical(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"x" * 100_000)
    b.write_bytes(b"x" * 100_000)
    assert _is_content_identical(a, b)
    b.write_bytes(b"x" * 99_999 + b"y")
    assert not _is_content_identical(a, b)
    b.write_bytes(b"x" * 50)  # size short-circuit
    assert not _is_content_identical(a, b)


def test_setup_verification_picks_lowest_format(tmp_path):
    d = tmp_path / "frags"
    d.mkdir()
    (d / "vid1.f299.Frag1").write_bytes(b"video")
    (d / "vid1.f140.Frag1").write_bytes(b"audio")
    result = _setup_verification("chan", d)
    assert result is not None
    backup, target_name = result
    assert target_name == "vid1.f140.Frag1"  # f140 sorts before f299
    assert backup.name == "vid1.f140.Frag1.bak"
    assert backup.exists()
    assert not (d / "vid1.f140.Frag1").exists()  # moved, not copied


def test_setup_verification_none_without_frag1(tmp_path):
    d = tmp_path / "frags"
    d.mkdir()
    (d / "vid1.f140.Frag2").write_bytes(b"x")
    assert _setup_verification("chan", d) is None


# --------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def tone_fragment(tmp_path_factory) -> Path:
    """A 2s AAC tone in MPEG-TS, stands in for one downloaded fragment."""
    path = tmp_path_factory.mktemp("fixtures") / "frag.ts"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:a",
            "aac",
            "-f",
            "mpegts",
            str(path),
        ],
        check=True,
    )
    return path


def dash_config(tmp_path, ytdlp: Path, **thresholds):
    defaults = {"fragment_seconds": 5.0, "lfs_gap_seconds": 10**9, "ytdlp_seconds": 30.0}
    defaults.update(thresholds)
    return make_config(
        tmp_path,
        server=ServerConfig(
            api_key="k",
            url="http://s.test",
            enabled=True,
            buffer_size_seconds=6.0,
            use_dash_for_youtube=True,
            stale_threshold=StaleThresholdConfig(**defaults),
        ),
        capture=CaptureConfig(yt_dlp_path=str(ytdlp), ffmpeg_path=ffmpeg),
    )


def fragment_writer_script(tmp_path: Path, fragment_dir: Path, tone: Path, body: str) -> Path:
    script = tmp_path / "fake-yt-dlp-dash"
    script.write_text(f'#!/bin/bash\nDIR="{fragment_dir}"\nTONE="{tone}"\nRUNS="{tmp_path}/runs"\necho run >> "$RUNS"\n{body}\n')
    script.chmod(0o755)
    return script


async def run_dash(config, state, should_stop=lambda: False):
    emitted = []

    async def emit(chunk):
        emitted.append(chunk)

    stats = await asyncio.wait_for(
        capture_stream_dash(config, STREAMER, state, INFO, emit, should_stop),
        timeout=60,
    )
    return stats, emitted


async def test_dash_accumulates_fragments_into_exact_chunks(tmp_path, tone_fragment):
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    fragment_dir = state.root / "fragments"
    script = fragment_writer_script(
        tmp_path,
        fragment_dir,
        tone_fragment,
        # Four 2s fragments: 3 fill a 6s chunk, the 4th flushes at exit.
        'for i in 1 2 3 4; do sleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag$i"; done\nexit 0',
    )
    config = dash_config(tmp_path, script)

    stats, emitted = await run_dash(config, state)

    assert stats.fell_behind is False
    assert len(emitted) == 2  # one full ~6s chunk + the ~2s flush
    first, second = emitted
    assert first.vod_accurate is True
    assert first.audio_start_time == START  # exact stream-zero anchor
    assert 5.5 <= first.duration <= 6.8
    assert second.audio_start_time == pytest.approx(START + first.duration, abs=0.01)
    assert first.path.exists() and len(first.pcm) > 0

    saved = json.loads((state.root / "dash_state.json").read_text())
    assert saved["stream_id"] == "vid1"
    assert saved["last_sequence"] == 4
    assert saved["current_stream_time"] == pytest.approx(START + first.duration + second.duration, abs=0.01)


async def test_dash_early_failure_falls_back(tmp_path, tone_fragment):
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    script = fragment_writer_script(tmp_path, state.root / "fragments", tone_fragment, "exit 1")
    config = dash_config(tmp_path, script)
    stats, emitted = await run_dash(config, state)
    assert stats.fell_behind is True
    assert emitted == []


async def test_dash_resume_skips_processed_sequences(tmp_path, tone_fragment):
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    fragment_dir = state.root / "fragments"
    fragment_dir.mkdir(parents=True)
    # Previous run: sequences 1-2 already emitted (4s of stream time), and
    # their fragment files are still on disk.
    shutil.copy(tone_fragment, fragment_dir / "vid1.f140.Frag1")
    shutil.copy(tone_fragment, fragment_dir / "vid1.f140.Frag2")
    _save_state(state.root / "dash_state.json", "vid1", 2, START + 4.0)

    script = fragment_writer_script(
        tmp_path,
        fragment_dir,
        tone_fragment,
        # yt-dlp re-downloads the identical Frag1 (continuity passes), then
        # produces the next sequence.
        'sleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag1"\nsleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag3"\nexit 0',
    )
    config = dash_config(tmp_path, script)

    stats, emitted = await run_dash(config, state)

    # Only sequence 3 is new; it flushes as a single ~2s chunk anchored at
    # the resumed stream time, no duplicate lines for sequences 1-2.
    assert len(emitted) == 1
    assert emitted[0].audio_start_time == pytest.approx(START + 4.0, abs=0.01)
    assert 1.5 <= emitted[0].duration <= 2.5
    saved = json.loads((state.root / "dash_state.json").read_text())
    assert saved["last_sequence"] == 3


async def test_dash_sequence_reset_keeps_stream_clock(tmp_path, tone_fragment):
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    fragment_dir = state.root / "fragments"
    fragment_dir.mkdir(parents=True)
    # Previous run's Frag1 has DIFFERENT bytes than what yt-dlp will now
    # fetch, the platform reset its fragment numbering.
    (fragment_dir / "vid1.f140.Frag1").write_bytes(b"old-stream-bytes")
    _save_state(state.root / "dash_state.json", "vid1", 2, START + 4.0)

    script = fragment_writer_script(
        tmp_path,
        fragment_dir,
        tone_fragment,
        # Every invocation writes fresh fragments; after the reset wipes the
        # dir and respawns, the second run repopulates it.
        'for i in 1 2 3; do sleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag$i"; done\nexit 0',
    )
    config = dash_config(tmp_path, script)

    stats, emitted = await run_dash(config, state)

    runs = (tmp_path / "runs").read_text().count("run")
    assert runs == 2  # original process + restart after the detected reset
    # Sequences restart at 1 but the stream clock is preserved: no time warp.
    assert len(emitted) >= 1
    assert emitted[0].audio_start_time == pytest.approx(START + 4.0, abs=0.01)
    saved = json.loads((state.root / "dash_state.json").read_text())
    assert saved["last_sequence"] == 3


async def test_dash_stall_watchdog_terminates(tmp_path, tone_fragment):
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    script = fragment_writer_script(tmp_path, state.root / "fragments", tone_fragment, "exec sleep 300")
    config = dash_config(tmp_path, script, ytdlp_seconds=1.0)
    started = time.monotonic()
    stats, emitted = await run_dash(config, state)
    assert emitted == []
    assert stats.fell_behind is False  # clean finish, not a live-edge switch
    assert time.monotonic() - started < 20


# ------------------------------------------------------- watcher selection


class _StrategyProbe:
    def __init__(self):
        self.calls = []

    def dash(self, stats):
        async def _run(config, streamer, state, info, emit, should_stop):
            self.calls.append("dash")
            return stats

        return _run

    async def live_edge(self, config, streamer, state, info, emit, should_stop):
        self.calls.append("live-edge")


async def test_watcher_strategy_selection(tmp_path, monkeypatch):
    from live_transcript_cloud_worker import watcher as watcher_module
    from live_transcript_cloud_worker.capture_dash import DashStats
    from live_transcript_cloud_worker.watcher import ChannelWatcher

    async def emit(chunk):
        pass

    def build_watcher(use_dash: bool):
        config = make_config(
            tmp_path,
            server=ServerConfig(
                api_key="k",
                url="http://s.test",
                enabled=True,
                use_dash_for_youtube=use_dash,
            ),
        )
        state = ChannelState("chan", tmp_path / "tmp" / "chan")
        return ChannelWatcher(
            config,
            STREAMER,
            state,
            server=None,
            prober=None,
            transcriber=None,
            uploader=None,
            stop_event=asyncio.Event(),
        )

    probe = _StrategyProbe()
    monkeypatch.setattr(watcher_module, "capture_stream", probe.live_edge)
    monkeypatch.setattr(watcher_module.capture_dash, "capture_stream_dash", probe.dash(DashStats()))
    monkeypatch.setattr(watcher_module.capture_dash, "resume_point", lambda s, i, d: time.time())

    yt_info = StreamInfo(url="https://www.youtube.com/watch?v=x", stream_id="x", is_live=True, start_time=int(time.time()))
    tw_info = StreamInfo(url="https://www.twitch.tv/x", stream_id="x", is_live=True, start_time=int(time.time()))

    # DASH on + YouTube -> DASH only (stream ended under DASH).
    await build_watcher(True)._capture(yt_info, emit)
    assert probe.calls == ["dash"]

    # DASH on + Twitch -> live-edge only, always.
    probe.calls.clear()
    await build_watcher(True)._capture(tw_info, emit)
    assert probe.calls == ["live-edge"]

    # DASH off -> live-edge for YouTube too.
    probe.calls.clear()
    await build_watcher(False)._capture(yt_info, emit)
    assert probe.calls == ["live-edge"]

    # DASH fell behind -> live-edge continues the stream.
    probe.calls.clear()
    monkeypatch.setattr(watcher_module.capture_dash, "capture_stream_dash", probe.dash(DashStats(fell_behind=True)))
    await build_watcher(True)._capture(yt_info, emit)
    assert probe.calls == ["dash", "live-edge"]

    # Pre-flight gap: resume point far behind live -> straight to live-edge.
    probe.calls.clear()
    monkeypatch.setattr(watcher_module.capture_dash, "resume_point", lambda s, i, d: time.time() - 10_000)
    await build_watcher(True)._capture(yt_info, emit)
    assert probe.calls == ["live-edge"]


# ----------------------------------------------------- review-fix regressions


@pytest.fixture(scope="module")
def video_only_fragment(tmp_path_factory) -> Path:
    """A 2s video-only TS (no audio track), a forced-through partial
    sequence whose audio fragment never downloaded."""
    path = tmp_path_factory.mktemp("fixtures") / "video_only.ts"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x64:rate=10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-f",
            "mpegts",
            str(path),
        ],
        check=True,
    )
    return path


async def test_container_duration(tone_fragment, tmp_path):
    from live_transcript_cloud_worker import audio

    assert await audio.container_duration(ffmpeg, tone_fragment) == pytest.approx(2.0, abs=0.3)
    garbage = tmp_path / "junk.ts"
    garbage.write_bytes(b"\x00" * 512)
    assert await audio.container_duration(ffmpeg, garbage) == 0.0


async def test_video_only_sequence_still_advances_vod_clock(tmp_path, tone_fragment, video_only_fragment):
    """A sequence with no decodable audio must advance the stream clock via
    container duration, or every later vodAccurate timestamp drifts early."""
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    fragment_dir = state.root / "fragments"
    script = fragment_writer_script(
        tmp_path,
        fragment_dir,
        tone_fragment,
        f'sleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag1"\n'
        f'sleep 0.2; cp "{video_only_fragment}" "$DIR/vid1.f140.Frag2"\n'
        f'sleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag3"\nexit 0',
    )
    config = dash_config(tmp_path, script)

    stats, emitted = await run_dash(config, state)

    saved = json.loads((state.root / "dash_state.json").read_text())
    # 2s audio + 2s video-only + 2s audio: the clock covers all ~6s.
    assert saved["current_stream_time"] - START == pytest.approx(6.0, abs=0.8)
    assert saved["last_sequence"] == 3


async def test_transient_merge_failure_is_retried(tmp_path, tone_fragment, monkeypatch):
    """One flaky merge (e.g. timeout under load) must be retried, not
    silently skipped with the sequence's audio lost forever."""
    from live_transcript_cloud_worker import capture_dash as cd

    real_merge = cd._merge_fragments
    failures = {"left": 1}

    async def flaky_merge(config, key, state, inputs, output):
        if failures["left"] > 0:
            failures["left"] -= 1
            return False
        return await real_merge(config, key, state, inputs, output)

    monkeypatch.setattr(cd, "_merge_fragments", flaky_merge)

    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    script = fragment_writer_script(
        tmp_path,
        state.root / "fragments",
        tone_fragment,
        'for i in 1 2; do sleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag$i"; done\nexit 0',
    )
    config = dash_config(tmp_path, script)

    stats, emitted = await run_dash(config, state)

    # Both 2s sequences survive: ~4s total, nothing dropped.
    total = sum(c.duration for c in emitted)
    assert total == pytest.approx(4.0, abs=0.6)
    assert failures["left"] == 0


async def test_fragment_dir_cleanup(tmp_path, tone_fragment):
    """Consumed fragments are deleted as capture goes (Frag1 kept for the
    continuity check); a cleanly ended stream drops the whole dir."""
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    fragment_dir = state.root / "fragments"

    # Mid-stream view: use the stall watchdog (not a clean end) so the dir
    # survives for inspection.
    script = fragment_writer_script(
        tmp_path,
        fragment_dir,
        tone_fragment,
        'for i in 1 2 3; do sleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag$i"; done\nexec sleep 300',
    )
    config = dash_config(tmp_path, script, ytdlp_seconds=2.0)
    stats, emitted = await run_dash(config, state)
    assert (fragment_dir / "vid1.f140.Frag1").exists()  # kept for verification
    assert not (fragment_dir / "vid1.f140.Frag2").exists()  # consumed: deleted
    assert not (fragment_dir / "vid1.f140.Frag3").exists()
    assert not list(fragment_dir.glob("merged_*.ts"))

    # Clean end: the whole dir goes away.
    script2 = fragment_writer_script(
        tmp_path,
        fragment_dir,
        tone_fragment,
        'sleep 0.2; cp "$TONE" "$DIR/vid1.f140.Frag4"\nexit 0',
    )
    config2 = dash_config(tmp_path, script2)
    await run_dash(config2, state)
    assert not fragment_dir.exists()
