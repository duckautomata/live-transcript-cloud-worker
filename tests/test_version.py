"""Version reporting: the startup banner and the heartbeat must show the
build's real version (the image's APP_VERSION), never a source constant."""

from __future__ import annotations

import asyncio

from live_transcript_cloud_worker import DEV_VERSION, app_version, build_time
from live_transcript_cloud_worker.heartbeat import heartbeat_loop


def test_version_comes_from_the_build_environment(monkeypatch):
    monkeypatch.setenv("APP_VERSION", "1.0.1")
    monkeypatch.setenv("BUILD_DATE", "2026-08-16T06:00:42Z")
    assert app_version() == "1.0.1"
    assert build_time() == "2026-08-16T06:00:42Z"


def test_version_falls_back_outside_a_build(monkeypatch):
    monkeypatch.delenv("APP_VERSION", raising=False)
    monkeypatch.delenv("BUILD_DATE", raising=False)
    assert app_version() == DEV_VERSION
    assert build_time() == "unknown"
    # An empty build arg is as good as unset.
    monkeypatch.setenv("APP_VERSION", "")
    assert app_version() == DEV_VERSION


async def test_heartbeat_reports_the_same_version(monkeypatch):
    monkeypatch.setenv("APP_VERSION", "1.0.1")
    monkeypatch.setenv("BUILD_DATE", "2026-08-16T06:00:42Z")
    posted = []

    class StubServer:
        async def post_status(self, version, build, keys, cookie_state=None, cookie_reason=""):
            posted.append((version, build, tuple(keys)))
            stop.set()

    stop = asyncio.Event()
    await heartbeat_loop(StubServer(), ["chan"], stop)

    assert posted == [(app_version(), build_time(), ("chan",))]
