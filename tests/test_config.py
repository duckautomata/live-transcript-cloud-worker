from __future__ import annotations

from pathlib import Path

import pytest

from live_transcript_cloud_worker.config import ConfigError, load_config

VALID = """
server:
  apiKey: secret
  url: https://server.example
  enabled: true
transcription:
  provider: deepgram
  deepgram:
    api_key: dg-key
streamers:
  - key: chan
    urls: [https://www.twitch.tv/chan]
    media_type: audio
"""


def write(tmp_path: Path, text: str) -> Path:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    path = cfg_dir / "config.yaml"
    path.write_text(text)
    return path


def test_valid_config_loads(tmp_path):
    config = load_config(write(tmp_path, VALID), tmp_path)
    assert config.server.api_key == "secret"
    assert config.server.url == "https://server.example"
    assert config.transcription.provider == "deepgram"
    assert config.active_streamers[0].key == "chan"
    assert config.server.buffer_size_seconds == 6.0  # default


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "config" / "nope.yaml", tmp_path)


def test_env_overrides_win(tmp_path, monkeypatch):
    monkeypatch.setenv("LT_SERVER_API_KEY", "env-key")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "env-dg")
    config = load_config(write(tmp_path, VALID), tmp_path)
    assert config.server.api_key == "env-key"
    assert config.transcription.options_for("deepgram")["api_key"] == "env-dg"


def test_missing_api_key_rejected(tmp_path):
    text = VALID.replace("apiKey: secret", "apiKey: ''")
    with pytest.raises(ConfigError, match="apiKey"):
        load_config(write(tmp_path, text), tmp_path)


def test_bad_provider_rejected(tmp_path):
    text = VALID.replace("provider: deepgram", "provider: openai")
    with pytest.raises(ConfigError, match="provider"):
        load_config(write(tmp_path, text), tmp_path)


def test_missing_provider_credentials_rejected(tmp_path):
    text = VALID.replace("api_key: dg-key", "api_key: ''")
    with pytest.raises(ConfigError, match="deepgram"):
        load_config(write(tmp_path, text), tmp_path)


def test_cloudflare_needs_account_and_token(tmp_path):
    text = VALID.replace("provider: deepgram", "provider: cloudflare")
    with pytest.raises(ConfigError, match="cloudflare"):
        load_config(write(tmp_path, text), tmp_path)


def test_fallback_must_differ(tmp_path):
    text = VALID.replace("provider: deepgram", "provider: deepgram\n  fallback_provider: deepgram")
    with pytest.raises(ConfigError, match="fallback"):
        load_config(write(tmp_path, text), tmp_path)


def test_bad_media_type_rejected(tmp_path):
    text = VALID.replace("media_type: audio", "media_type: hologram")
    with pytest.raises(ConfigError, match="media_type"):
        load_config(write(tmp_path, text), tmp_path)


def test_duplicate_keys_rejected(tmp_path):
    text = (
        VALID
        + """
  - key: chan
    urls: [https://example.com/x]
"""
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write(tmp_path, text), tmp_path)


def test_unknown_key_rejected(tmp_path):
    text = (
        VALID
        + """
capture:
  yt_dlp_pathh: /oops
"""
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, text), tmp_path)


def test_local_mode_needs_no_key_or_url(tmp_path):
    text = """
server:
  enabled: false
transcription:
  provider: deepgram
  deepgram:
    api_key: dg
streamers:
  - key: chan
    urls: [https://www.twitch.tv/chan]
"""
    config = load_config(write(tmp_path, text), tmp_path)
    assert not config.server.enabled


def test_example_config_parses(tmp_path, monkeypatch):
    example = Path(__file__).resolve().parent.parent / "config" / "example.yaml"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
    monkeypatch.setenv("LT_SERVER_API_KEY", "x")
    config = load_config(example, tmp_path)
    assert config.transcription.provider == "deepgram"
    assert config.server.stale_threshold.ytdlp_seconds == 180


def test_unknown_provider_section_rejected(tmp_path):
    text = VALID.replace(
        "provider: deepgram",
        "provider: deepgram\n  acme:\n    api_key: x",
    )
    with pytest.raises(ConfigError, match="no such provider"):
        load_config(write(tmp_path, text), tmp_path)


def test_unknown_provider_option_rejected(tmp_path):
    text = VALID.replace("api_key: dg-key", "api_key: dg-key\n    modle: typo")
    with pytest.raises(ConfigError, match="unknown option"):
        load_config(write(tmp_path, text), tmp_path)


def test_provider_env_overrides_populate_options(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env-acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "env-tok")
    config = load_config(write(tmp_path, VALID), tmp_path)
    assert config.transcription.options_for("cloudflare") == {
        "account_id": "env-acct",
        "api_token": "env-tok",
    }


def test_streamer_live_from_start_rejected(tmp_path):
    # Per-streamer live_from_start was removed with Twitch LFS capture;
    # DASH is toggled by server.use_dash_for_youtube instead.
    text = VALID.replace("media_type: audio", "media_type: audio\n    live_from_start: true")
    with pytest.raises(ConfigError, match="unknown key.*streamers"):
        load_config(write(tmp_path, text), tmp_path)


def test_dash_config_accepted(tmp_path):
    text = VALID.replace(
        "url: https://server.example",
        "url: https://server.example\n  use_dash_for_youtube: true\n  stale_threshold:\n    fragment_seconds: 30\n    lfs_gap_seconds: 300",
    )
    config = load_config(write(tmp_path, text), tmp_path)
    assert config.server.use_dash_for_youtube
    assert config.server.stale_threshold.fragment_seconds == 30
    assert config.server.stale_threshold.lfs_gap_seconds == 300


def test_dash_with_cookies_rejected(tmp_path):
    text = VALID.replace(
        "url: https://server.example",
        "url: https://server.example\n  use_dash_for_youtube: true\n  cookies:\n    enabled: true",
    )
    with pytest.raises(ConfigError, match="use_dash_for_youtube must be turned off"):
        load_config(write(tmp_path, text), tmp_path)


def test_unknown_top_level_and_server_keys_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(write(tmp_path, VALID + "\nextras: {}\n"), tmp_path)
    text = VALID.replace("apiKey: secret", "apiKey: secret\n  buffer_size_secondss: 12")
    with pytest.raises(ConfigError, match="unknown key.*server"):
        load_config(write(tmp_path, text), tmp_path)


def test_blank_scalars_fall_back_to_defaults(tmp_path):
    text = VALID.replace(
        "provider: deepgram",
        "provider: deepgram\n  language:\n  fallback_provider:",
    ).replace("api_key: dg-key", "api_key: dg-key\n    model:")
    config = load_config(write(tmp_path, text), tmp_path)
    assert config.transcription.language == "en"
    assert config.transcription.fallback_provider == ""
    # A blank option key means "use the provider's default", not None.
    assert "model" not in config.transcription.options_for("deepgram")


def test_garbage_numeric_scalars_raise_config_error(tmp_path):
    text = VALID.replace("provider: deepgram", "provider: deepgram\n  max_retries: lots")
    with pytest.raises(ConfigError, match="max_retries must be a number"):
        load_config(write(tmp_path, text), tmp_path)
    text = VALID.replace(
        "url: https://server.example",
        "url: https://server.example\n  channel_polling:\n    interval_seconds: fast",
    )
    with pytest.raises(ConfigError, match="interval_seconds must be a number"):
        load_config(write(tmp_path, text), tmp_path)
