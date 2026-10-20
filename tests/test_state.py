from __future__ import annotations

import json
from pathlib import Path

import pytest

from live_transcript_cloud_worker.models import Line, MediaType, Segment
from live_transcript_cloud_worker.state import ChannelState


def line(i: int, ts: int = 1000) -> Line:
    return Line(id=i, timestamp=ts + i * 6, segments=(Segment(ts + i * 6, f"line {i}"),), vod_accurate=True)


@pytest.fixture
def state(tmp_path: Path) -> ChannelState:
    return ChannelState("chan", tmp_path / "chan")


def test_fresh_state(state: ChannelState):
    assert state.stream_id == ""
    assert state.next_line_id == 0
    assert state.transcript == []


def test_append_and_reload(state: ChannelState, tmp_path: Path):
    state.start_new_stream("abc", "Title", 1000, MediaType.AUDIO)
    state.append_line(line(0))
    state.append_line(line(1))

    reloaded = ChannelState("chan", tmp_path / "chan")
    assert reloaded.stream_id == "abc"
    assert reloaded.next_line_id == 2
    assert reloaded.transcript[1].segments[0].text == "line 1"
    assert reloaded.is_live


def test_append_rejects_gap(state: ChannelState):
    state.start_new_stream("abc", "T", 1000, MediaType.NONE)
    state.append_line(line(0))
    with pytest.raises(ValueError):
        state.append_line(line(2))


def test_torn_final_record_is_truncated(state: ChannelState, tmp_path: Path):
    state.start_new_stream("abc", "T", 1000, MediaType.NONE)
    state.append_line(line(0))
    state.append_line(line(1))
    path = tmp_path / "chan" / "transcript.jsonl"
    with open(path, "ab") as f:
        f.write(b'{"id": 2, "timestamp": 101')  # crash mid-write

    reloaded = ChannelState("chan", tmp_path / "chan")
    assert reloaded.next_line_id == 2
    # The torn tail is gone; a subsequent append stays consistent.
    reloaded.append_line(line(2))
    records = [json.loads(rec) for rec in path.read_bytes().splitlines() if rec.strip()]
    assert [r["id"] for r in records] == [0, 1, 2]


def test_id_discontinuity_truncates(state: ChannelState, tmp_path: Path):
    state.start_new_stream("abc", "T", 1000, MediaType.NONE)
    state.append_line(line(0))
    path = tmp_path / "chan" / "transcript.jsonl"
    with open(path, "ab") as f:
        f.write(json.dumps(line(5).to_wire()).encode() + b"\n")

    reloaded = ChannelState("chan", tmp_path / "chan")
    assert reloaded.next_line_id == 1


def test_new_stream_resets_everything(state: ChannelState, tmp_path: Path):
    state.start_new_stream("abc", "T", 1000, MediaType.AUDIO)
    state.append_line(line(0))
    src = tmp_path / "chan" / "segments" / "x.ts"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"ts-bytes")
    state.enqueue_media("abc", 0, src)
    assert len(state.pending_media()) == 1

    state.start_new_stream("def", "T2", 2000, MediaType.AUDIO)
    assert state.next_line_id == 0
    assert state.pending_media() == []
    assert state.transcript == []


def test_same_stream_refresh_keeps_transcript(state: ChannelState):
    state.start_new_stream("abc", "T", 1000, MediaType.AUDIO)
    state.append_line(line(0))
    state.refresh_stream("New title", 1000)
    assert state.next_line_id == 1
    assert state.stream_title == "New title"


def test_media_queue_roundtrip(state: ChannelState, tmp_path: Path):
    src = tmp_path / "chan" / "segments" / "chunk000001.ts"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"data")
    dest = state.enqueue_media("abc", 7, src)
    assert not src.exists()
    assert dest.exists()

    pending = state.pending_media()
    assert len(pending) == 1
    assert pending[0].stream_id == "abc"
    assert pending[0].line_id == 7


def test_pending_media_ignores_tmp_and_foreign(state: ChannelState):
    (state.queue_dir / "abc__000001.ts.tmp").write_bytes(b"partial")
    (state.queue_dir / "junk.bin").write_bytes(b"junk")
    (state.queue_dir / "abc__000002.ts").write_bytes(b"good")
    pending = state.pending_media()
    assert [p.line_id for p in pending] == [2]


def test_sync_payload_shape(state: ChannelState):
    state.start_new_stream("abc", "T", 1000, MediaType.AUDIO)
    state.append_line(line(0))
    payload = state.sync_payload()
    assert payload["streamId"] == "abc"
    assert payload["startTime"] == "1000"  # string, per wire format
    assert payload["isLive"] is True
    assert payload["transcript"][0]["id"] == 0
    assert payload["transcript"][0]["mediaAvailable"] is False


def test_corrupt_meta_starts_fresh(tmp_path: Path):
    root = tmp_path / "chan"
    root.mkdir()
    (root / "meta.json").write_bytes(b"{not json")
    state = ChannelState("chan", root)
    assert state.stream_id == ""


def test_remove_legacy_dash_files(tmp_path: Path, caplog):
    """The removed DASH capture's leftovers go at boot; nothing else does."""
    root = tmp_path / "chan"
    fragments = root / "fragments"
    fragments.mkdir(parents=True)
    (fragments / "vid1.f140.Frag7").write_bytes(b"frag")
    (fragments / "merged_7.ts").write_bytes(b"ts")
    (root / "dash_state.json").write_text('{"stream_id": "vid1", "last_sequence": 7}')
    (root / "youtube_timeline_state.json").write_text('{"stream_id": "vid1", "next_sequence": 8}')
    state = ChannelState("chan", root)

    with caplog.at_level("INFO"):
        state.remove_legacy_dash_files()

    assert not fragments.exists()
    assert not (root / "dash_state.json").exists()
    assert (root / "youtube_timeline_state.json").exists()
    assert state.queue_dir.is_dir() and state.segments_dir.is_dir()
    assert "removed leftovers of the old DASH capture: fragments/, dash_state.json" in caplog.text

    # Nothing left: a no-op that logs nothing.
    caplog.clear()
    state.remove_legacy_dash_files()
    assert "DASH" not in caplog.text
