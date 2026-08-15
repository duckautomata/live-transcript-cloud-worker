"""Pipeline ordering invariants: gapless IDs, serialized submission,
409 -> sync recovery, media-after-line, failure containment."""

from __future__ import annotations

import asyncio
from pathlib import Path

from conftest import make_config

from live_transcript_cloud_worker.config import StreamerConfig
from live_transcript_cloud_worker.models import Chunk, MediaType
from live_transcript_cloud_worker.pipeline import StreamPipeline
from live_transcript_cloud_worker.server_client import LineResult
from live_transcript_cloud_worker.state import ChannelState


class FakeServer:
    def __init__(self):
        self.lines = []  # (stream_id, wire) in arrival order
        self.syncs = []
        self.line_results: list[LineResult] = []
        self.sync_ok = True

    async def post_line(self, channel, stream_id, line):
        self.lines.append((stream_id, line.to_wire()))
        if self.line_results:
            return self.line_results.pop(0)
        return LineResult.OK

    async def sync(self, channel, payload):
        self.syncs.append(payload)
        return self.sync_ok


class FakeTranscriber:
    def __init__(self, results=None):
        self.results = results or {}
        self.in_flight = 0
        self.max_in_flight = 0

    async def transcribe(self, wav):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        pcm_len = len(wav) - 44  # strip the WAV header the pipeline added
        return self.results.get(pcm_len, [(0.0, f"text-{pcm_len}")])


class FakeUploader:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, key, stream_id, line_id, path):
        self.enqueued.append((key, stream_id, line_id, path))


def make_chunk(tmp_path: Path, i: int, media_type=MediaType.AUDIO, stream_id="abc") -> Chunk:
    path = tmp_path / f"chunk{i:06d}.ts"
    path.write_bytes(b"ts")
    return Chunk(
        key="chan",
        stream_id=stream_id,
        path=path,
        audio_start_time=1000.0 + i * 6,
        duration=6.0,
        vod_accurate=True,
        media_type=media_type,
        pcm=bytes(i + 1),  # distinct lengths key the fake transcriber
    )


def build(tmp_path, media_type=MediaType.AUDIO, transcriber=None, server=None):
    config = make_config(tmp_path)
    streamer = StreamerConfig(key="chan", urls=("u",), media_type=media_type)
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    state.start_new_stream("abc", "T", 1000, media_type)
    server = server or FakeServer()
    transcriber = transcriber or FakeTranscriber()
    uploader = FakeUploader()
    pipeline = StreamPipeline(config, streamer, state, server, transcriber, uploader)
    return pipeline, state, server, uploader


async def run_pipeline(pipeline, chunks):
    task = asyncio.create_task(pipeline.run())
    for chunk in chunks:
        await pipeline.emit(chunk)
    await pipeline.finish()
    await task


async def test_lines_are_gapless_and_ordered(tmp_path):
    pipeline, state, server, _ = build(tmp_path)
    await run_pipeline(pipeline, [make_chunk(tmp_path, i) for i in range(5)])
    assert [wire["id"] for _, wire in server.lines] == [0, 1, 2, 3, 4]
    assert state.next_line_id == 5
    # Timestamps are ints derived from chunk starts.
    assert [wire["timestamp"] for _, wire in server.lines] == [1000, 1006, 1012, 1018, 1024]


async def test_conflict_triggers_full_sync_and_continues(tmp_path):
    server = FakeServer()
    server.line_results = [LineResult.OK, LineResult.CONFLICT, LineResult.OK]
    pipeline, state, server, _ = build(tmp_path, server=server)
    await run_pipeline(pipeline, [make_chunk(tmp_path, i) for i in range(3)])
    assert len(server.syncs) == 1
    # The sync payload already contains the conflicted line (id=1).
    assert [rec["id"] for rec in server.syncs[0]["transcript"]] == [0, 1]
    # No renumbering afterwards.
    assert [wire["id"] for _, wire in server.lines] == [0, 1, 2]


async def test_failed_line_kept_locally_and_ids_advance(tmp_path):
    server = FakeServer()
    server.line_results = [LineResult.FAILED, LineResult.OK]
    pipeline, state, server, _ = build(tmp_path, server=server)
    await run_pipeline(pipeline, [make_chunk(tmp_path, i) for i in range(2)])
    assert state.next_line_id == 2
    assert [wire["id"] for _, wire in server.lines] == [0, 1]


async def test_media_enqueued_after_line_with_matching_ids(tmp_path):
    pipeline, state, server, uploader = build(tmp_path)
    await run_pipeline(pipeline, [make_chunk(tmp_path, i) for i in range(3)])
    assert [(sid, lid) for _, sid, lid, _ in uploader.enqueued] == [("abc", 0), ("abc", 1), ("abc", 2)]
    for _, _, _, path in uploader.enqueued:
        assert path.exists()
        assert path.parent == state.queue_dir


async def test_media_type_none_deletes_chunk_files(tmp_path):
    pipeline, _, _, uploader = build(tmp_path, media_type=MediaType.NONE)
    chunks = [make_chunk(tmp_path, i, media_type=MediaType.NONE) for i in range(2)]
    await run_pipeline(pipeline, chunks)
    assert uploader.enqueued == []
    for chunk in chunks:
        assert not chunk.path.exists()


async def test_transcription_failure_still_emits_line(tmp_path):
    class NoneTranscriber:
        async def transcribe(self, wav):
            return None

    pipeline, state, server, _ = build(tmp_path, transcriber=NoneTranscriber())
    await run_pipeline(pipeline, [make_chunk(tmp_path, 0)])
    assert len(server.lines) == 1
    assert server.lines[0][1]["segments"] == []
    assert state.next_line_id == 1


async def test_transcriptions_overlap_but_submission_is_ordered(tmp_path):
    transcriber = FakeTranscriber()
    pipeline, _, server, _ = build(tmp_path, transcriber=transcriber)
    await run_pipeline(pipeline, [make_chunk(tmp_path, i) for i in range(6)])
    assert transcriber.max_in_flight >= 2  # concurrency actually used
    assert [wire["id"] for _, wire in server.lines] == list(range(6))


async def test_empty_segment_text_is_dropped_but_line_emitted(tmp_path):
    transcriber = FakeTranscriber(results={1: [(0.0, "   "), (1.0, "keep")]})
    pipeline, _, server, _ = build(tmp_path, transcriber=transcriber)
    await run_pipeline(pipeline, [make_chunk(tmp_path, 0)])
    assert server.lines[0][1]["segments"] == [{"timestamp": 1001, "text": "keep"}]
