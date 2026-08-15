from __future__ import annotations

from pathlib import Path

import pytest

from live_transcript_cloud_worker.config import (
    CaptureConfig,
    Config,
    ServerConfig,
    StreamerConfig,
    TranscriptionConfig,
)
from live_transcript_cloud_worker.models import MediaType


def make_config(base_dir: Path, **kwargs) -> Config:
    server = kwargs.pop(
        "server",
        ServerConfig(api_key="test-key", url="http://server.test", enabled=True),
    )
    transcription = kwargs.pop(
        "transcription",
        TranscriptionConfig(
            provider="deepgram",
            provider_options={
                "deepgram": {"api_key": "dg-key"},
                "cloudflare": {"account_id": "acct", "api_token": "tok"},
            },
        ),
    )
    streamers = kwargs.pop(
        "streamers",
        (
            StreamerConfig(
                key="chan",
                urls=("https://www.youtube.com/@x/live",),
                media_type=MediaType.AUDIO,
            ),
        ),
    )
    return Config(
        server=server,
        capture=kwargs.pop("capture", CaptureConfig()),
        transcription=transcription,
        streamers=streamers,
        id_blacklist=kwargs.pop("id_blacklist", ()),
        base_dir=base_dir,
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return make_config(tmp_path)
