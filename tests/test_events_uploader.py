"""Events listener signal semantics and media uploader fairness/durability."""

from __future__ import annotations

import asyncio
from pathlib import Path

from conftest import make_config

from live_transcript_cloud_worker.events import EventsListener
from live_transcript_cloud_worker.server_client import MediaResult
from live_transcript_cloud_worker.state import ChannelState
from live_transcript_cloud_worker.uploader import MediaUploader


class StubWatcher:
    def __init__(self, key: str):
        class S:  # minimal streamer stand-in
            pass

        self.streamer = S()
        self.streamer.key = key
        self.restart_event = asyncio.Event()
        self.incoming_event = asyncio.Event()


class EventsServer:
    def __init__(self, script, stop_event):
        self.script = list(script)
        self.stop_event = stop_event
        self.calls = []
        self.acks = []
        self.restart_pending = {}

    async def get_events(self, channels, since, wait):
        self.calls.append(since)
        if not self.script:
            self.stop_event.set()
            return {}, since
        return self.script.pop(0)

    async def ack_restart(self, key):
        self.acks.append(key)
        return True

    async def get_restart(self, key):
        return self.restart_pending.get(key, False)


async def test_events_cursor_echoed_and_signals_fanned_out(tmp_path):
    stop = asyncio.Event()
    server = EventsServer(
        [({"chan": ["incoming", "restart"]}, 111), ({}, 111)],
        stop,
    )
    watcher = StubWatcher("chan")
    config = make_config(tmp_path)
    listener = EventsListener(config, server, {"chan": watcher}, stop)
    await asyncio.wait_for(listener.run(), timeout=5)

    assert watcher.incoming_event.is_set()
    assert watcher.restart_event.is_set()
    assert server.acks == ["chan"]
    # The cursor from the first response is echoed on the next call.
    assert server.calls[0] == 0
    assert server.calls[1] == 111


async def test_restart_already_handling_not_reacked(tmp_path):
    stop = asyncio.Event()
    server = EventsServer([({"chan": ["restart"]}, 5)], stop)
    watcher = StubWatcher("chan")
    watcher.restart_event.set()  # already being handled
    config = make_config(tmp_path)
    listener = EventsListener(config, server, {"chan": watcher}, stop)
    await asyncio.wait_for(listener.run(), timeout=5)
    assert server.acks == []  # must not wipe a freshly POSTed request


class RecordingServer:
    def __init__(self, results=None):
        self.uploads = []
        self.results = results or {}

    async def upload_media(self, key, stream_id, line_id, path):
        self.uploads.append((key, stream_id, line_id))
        await asyncio.sleep(0.005)
        return self.results.get((key, line_id), MediaResult.OK)


async def make_job_file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"ts")
    return path


async def test_uploader_deletes_file_after_success_and_drop(tmp_path):
    config = make_config(tmp_path)
    server = RecordingServer(results={("a", 1): MediaResult.DROP})
    uploader = MediaUploader(config, server)
    uploader.start()
    ok_file = await make_job_file(tmp_path, "ok.ts")
    drop_file = await make_job_file(tmp_path, "drop.ts")
    uploader.enqueue("a", "s", 0, ok_file)
    uploader.enqueue("a", "s", 1, drop_file)
    assert await uploader.drain(5)
    await uploader.stop()
    assert not ok_file.exists()
    assert not drop_file.exists()
    assert len(server.uploads) == 2


async def test_uploader_round_robin_across_channels(tmp_path):
    config = make_config(tmp_path)
    server = RecordingServer()
    uploader = MediaUploader(config, server)
    # Backlog for channel a, one item for channel b; b must not wait for
    # a's whole backlog. Use a single worker to make ordering deterministic.
    jobs = []
    for i in range(3):
        jobs.append(("a", await make_job_file(tmp_path, f"a{i}.ts"), i))
    jobs.append(("b", await make_job_file(tmp_path, "b0.ts"), 0))
    for key, path, line_id in jobs:
        uploader.enqueue(key, "s", line_id, path)
    # Single worker for determinism.
    uploader._workers = [asyncio.create_task(uploader._worker())]
    assert await uploader.drain(5)
    await uploader.stop()
    order = [(k, lid) for k, _, lid in server.uploads]
    # b's single job runs after a's first, not after a's entire backlog.
    assert order.index(("b", 0)) == 1


async def test_uploader_rescan_reenqueues_disk_queue(tmp_path):
    config = make_config(tmp_path)
    state = ChannelState("chan", tmp_path / "tmp" / "chan")
    (state.queue_dir / "abc__000004.ts").write_bytes(b"x")
    (state.queue_dir / "abc__000002.ts").write_bytes(b"x")
    server = RecordingServer()
    uploader = MediaUploader(config, server)
    assert uploader.rescan({"chan": state}) == 2
    uploader.start()
    assert await uploader.drain(5)
    await uploader.stop()
    # Ordered by line id.
    assert [u[2] for u in server.uploads] == [2, 4]


async def test_uploader_missing_file_skipped(tmp_path):
    config = make_config(tmp_path)
    server = RecordingServer()
    uploader = MediaUploader(config, server)
    uploader.start()
    uploader.enqueue("a", "s", 0, tmp_path / "vanished.ts")
    assert await uploader.drain(5)
    await uploader.stop()
    assert server.uploads == []
