"""Configuration: loaded once at startup into immutable dataclasses.

Schema follows docs/05-configuration.md, with a ``transcription`` section
replacing the local-whisper one: transcription is delegated to a remote
provider (Cloudflare Workers AI or Deepgram), so no GPU settings exist.

Secrets may come from the YAML or from environment variables (env wins):
    LT_SERVER_API_KEY       -> server.apiKey
    DEEPGRAM_API_KEY        -> transcription.deepgram.api_key
    CLOUDFLARE_API_TOKEN    -> transcription.cloudflare.api_token
    CLOUDFLARE_ACCOUNT_ID   -> transcription.cloudflare.account_id
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .models import MediaType, valid_stream_id
from .providers import get_provider_class, registered_names


class ConfigError(Exception):
    """Raised when the config file is missing, unparseable, or invalid."""


@dataclass(frozen=True)
class IncomingPollingConfig:
    enabled: bool = False
    # Queue poll cadence without events polling; with it, the maximum age of
    # the worker's copy of the queue before a probe is spent on a URL.
    interval_seconds: float = 30.0
    # Consecutive trusted confirmed-offline probes before a queued URL is removed.
    offline_delete_threshold: int = 2
    # Consecutive probes that could not tell whether the stream is live
    # before the worker gives up on a queued URL. 0 = never.
    inconclusive_delete_threshold: int = 10
    # ...and the streak must also have lasted at least this long, so a short
    # worker-wide hiccup cannot drain every channel's queue. 0 = count only.
    inconclusive_min_span_seconds: float = 3600.0


@dataclass(frozen=True)
class EventsPollingConfig:
    enabled: bool = True
    wait_seconds: int = 25
    fallback_interval_seconds: float = 300.0


@dataclass(frozen=True)
class ChannelPollingConfig:
    interval_seconds: float = 60.0
    max_interval_seconds: float = 9000.0
    pre_scheduled_buffer_seconds: float = 300.0


@dataclass(frozen=True)
class CookiesConfig:
    enabled: bool = False
    check_filename: str = "cookies.txt"
    download_filename: str = "cookies.txt"
    # Probe cadence for YouTube URLs while yt-dlp reports the jar is dead.
    degraded_probe_interval_seconds: float = 600.0


@dataclass(frozen=True)
class StaleThresholdConfig:
    # DASH only: wait this long for a sequence's tracks to complete before
    # emitting it with partial data.
    fragment_seconds: float = 60.0
    # DASH only: falling this far behind live abandons catch-up and switches
    # to live-edge capture for the rest of the stream.
    lfs_gap_seconds: float = 600.0
    # No new segment/fragment for this long => terminate a wedged yt-dlp.
    ytdlp_seconds: float = 180.0


@dataclass(frozen=True)
class ServerConfig:
    api_key: str = ""
    url: str = ""
    enabled: bool = True
    buffer_size_seconds: float = 6.0
    # YouTube only: capture with the fragment-based DASH live-from-start
    # strategy (exact vodAccurate timestamps, catches the stream from second
    # zero). Must be OFF when cookies are enabled. Twitch always uses
    # live-edge regardless of this setting.
    use_dash_for_youtube: bool = False
    media_upload_concurrency: int = 3
    incoming_polling: IncomingPollingConfig = field(default_factory=IncomingPollingConfig)
    events_polling: EventsPollingConfig = field(default_factory=EventsPollingConfig)
    channel_polling: ChannelPollingConfig = field(default_factory=ChannelPollingConfig)
    cookies: CookiesConfig = field(default_factory=CookiesConfig)
    stale_threshold: StaleThresholdConfig = field(default_factory=StaleThresholdConfig)


@dataclass(frozen=True)
class CaptureConfig:
    yt_dlp_path: str = "yt-dlp"
    ffmpeg_path: str = "ffmpeg"
    # Live-edge timestamp anchoring: mtime - duration - live_latency_seconds.
    live_latency_seconds: float = 1.0
    # Process-wide cap on concurrent yt-dlp -j probes (docs/07 §7).
    probe_concurrency: int = 4
    probe_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class TranscriptionConfig:
    provider: str = "deepgram"
    # Optional second provider tried when the primary exhausts its retries.
    fallback_provider: str = ""
    language: str = "en"
    # Per-channel in-flight transcription requests; results are re-ordered
    # before line submission so ordering invariants hold regardless.
    concurrency: int = 2
    request_timeout_seconds: float = 30.0
    max_retries: int = 4
    # Chunks measured shorter than this are dropped before an ID is assigned.
    min_chunk_seconds: float = 0.5
    # Per-provider option blocks, keyed by registered provider name. Any
    # mapping under ``transcription:`` besides the scalar keys above lands
    # here verbatim; providers validate their own options.
    provider_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def options_for(self, name: str) -> dict[str, Any]:
        return dict(self.provider_options.get(name, {}))


@dataclass(frozen=True)
class StreamerConfig:
    key: str
    urls: tuple[str, ...] = ()
    active: bool = True
    media_type: MediaType = MediaType.NONE


@dataclass(frozen=True)
class Config:
    server: ServerConfig
    capture: CaptureConfig
    transcription: TranscriptionConfig
    streamers: tuple[StreamerConfig, ...]
    id_blacklist: tuple[str, ...]
    base_dir: Path

    @property
    def active_streamers(self) -> tuple[StreamerConfig, ...]:
        return tuple(s for s in self.streamers if s.active)

    def channel_dir(self, key: str) -> Path:
        return self.base_dir / "tmp" / key

    def cookies_file(self, which: str) -> Path | None:
        """Path to the cookies file for 'check' or 'download', or None if disabled."""
        cookies = self.server.cookies
        if not cookies.enabled:
            return None
        name = cookies.check_filename if which == "check" else cookies.download_filename
        return self.base_dir / name


def _section(raw: Any, name: str) -> dict[str, Any]:
    value = (raw or {}).get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{name}' must be a mapping")
    return value


def _number(section: dict[str, Any], key: str, default: float, cast: type, path: str) -> Any:
    """Read an optional numeric key; blank means default, garbage is loud."""
    value = section.get(key)
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{path}.{key} must be a number, got {value!r}") from None


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    """Build a flat dataclass from a dict, rejecting unknown keys loudly and
    coercing numeric fields with a friendly error."""
    dc_fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    unknown = set(data) - set(dc_fields)
    if unknown:
        raise ConfigError(f"unknown key(s) in {path}: {', '.join(sorted(unknown))}")
    coerced = dict(data)
    for name, field_info in dc_fields.items():
        if name not in coerced:
            continue
        if field_info.type in ("int", "float"):
            cast = int if field_info.type == "int" else float
            coerced[name] = _number(coerced, name, field_info.default, cast, path)
    return cls(**coerced)


def load_config(config_path: Path, base_dir: Path | None = None) -> Config:
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")

    base = (base_dir or config_path.parent.parent).resolve()

    unknown_top = set(raw) - {"server", "capture", "transcription", "streamers", "id_blacklist"}
    if unknown_top:
        raise ConfigError(f"unknown top-level key(s): {', '.join(sorted(unknown_top))}")

    srv = _section(raw, "server")
    known_server_keys = {
        "apiKey",
        "url",
        "enabled",
        "buffer_size_seconds",
        "use_dash_for_youtube",
        "media_upload_concurrency",
        "incoming_polling",
        "events_polling",
        "channel_polling",
        "cookies",
        "stale_threshold",
    }
    unknown_server = set(srv) - known_server_keys
    if unknown_server:
        raise ConfigError(f"unknown key(s) in server: {', '.join(sorted(unknown_server))}")
    server = ServerConfig(
        api_key=os.environ.get("LT_SERVER_API_KEY") or str(srv.get("apiKey") or ""),
        url=str(srv.get("url") or "").rstrip("/"),
        enabled=bool(srv.get("enabled", True)),
        buffer_size_seconds=_number(srv, "buffer_size_seconds", 6.0, float, "server"),
        use_dash_for_youtube=bool(srv.get("use_dash_for_youtube", False)),
        media_upload_concurrency=_number(srv, "media_upload_concurrency", 3, int, "server"),
        incoming_polling=_build(IncomingPollingConfig, _section(srv, "incoming_polling"), "server.incoming_polling"),
        events_polling=_build(EventsPollingConfig, _section(srv, "events_polling"), "server.events_polling"),
        channel_polling=_build(ChannelPollingConfig, _section(srv, "channel_polling"), "server.channel_polling"),
        cookies=_build(CookiesConfig, _section(srv, "cookies"), "server.cookies"),
        stale_threshold=_build(StaleThresholdConfig, _section(srv, "stale_threshold"), "server.stale_threshold"),
    )

    capture = _build(CaptureConfig, _section(raw, "capture"), "capture")

    tr = _section(raw, "transcription")
    scalar_keys = {
        "provider",
        "fallback_provider",
        "language",
        "concurrency",
        "request_timeout_seconds",
        "max_retries",
        "min_chunk_seconds",
    }
    provider_options: dict[str, dict[str, Any]] = {}
    for key, value in tr.items():
        if key in scalar_keys:
            continue
        if isinstance(value, dict) or value is None:
            provider_options[str(key)] = {k: v for k, v in (value or {}).items() if v is not None}
        else:
            raise ConfigError(f"unknown key in transcription: {key!r} (provider sections must be mappings)")
    # Environment variables override file-sourced credentials, per provider.
    for name in registered_names():
        cls = get_provider_class(name)
        assert cls is not None
        for option_key, env_var in cls.env_overrides.items():
            value = os.environ.get(env_var)
            if value:
                provider_options.setdefault(name, {})[option_key] = value
    transcription = TranscriptionConfig(
        provider=str(tr.get("provider") or "deepgram").lower(),
        fallback_provider=str(tr.get("fallback_provider") or "").lower(),
        language=str(tr.get("language") or "en"),
        concurrency=_number(tr, "concurrency", 2, int, "transcription"),
        request_timeout_seconds=_number(tr, "request_timeout_seconds", 30.0, float, "transcription"),
        max_retries=_number(tr, "max_retries", 4, int, "transcription"),
        min_chunk_seconds=_number(tr, "min_chunk_seconds", 0.5, float, "transcription"),
        provider_options=provider_options,
    )

    streamers_raw = raw.get("streamers") or []
    if not isinstance(streamers_raw, list):
        raise ConfigError("'streamers' must be a list")
    streamers = []
    known_streamer_keys = {"key", "urls", "active", "media_type"}
    for i, entry in enumerate(streamers_raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"streamers[{i}] must be a mapping")
        unknown_streamer = set(entry) - known_streamer_keys
        if unknown_streamer:
            raise ConfigError(f"unknown key(s) in streamers[{i}]: {', '.join(sorted(unknown_streamer))}")
        try:
            media_type = MediaType(str(entry.get("media_type", "none")))
        except ValueError:
            raise ConfigError(f"streamers[{i}].media_type must be one of none|audio|video, got {entry.get('media_type')!r}") from None
        streamers.append(
            StreamerConfig(
                key=str(entry.get("key", "")),
                urls=tuple(str(u) for u in (entry.get("urls") or [])),
                active=bool(entry.get("active", True)),
                media_type=media_type,
            )
        )

    config = Config(
        server=server,
        capture=capture,
        transcription=transcription,
        streamers=tuple(streamers),
        id_blacklist=tuple(str(x) for x in (raw.get("id_blacklist") or [])),
        base_dir=base,
    )
    _validate(config)
    return config


def _validate(config: Config) -> None:
    problems: list[str] = []
    server = config.server

    if server.enabled:
        parsed = urlparse(server.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            problems.append(f"server.url must be an absolute http(s) URL, got {server.url!r}")
        if not server.api_key:
            problems.append("server.apiKey (or LT_SERVER_API_KEY) is required when server.enabled")
    if server.buffer_size_seconds <= 0:
        problems.append("server.buffer_size_seconds must be > 0")
    if not (0 <= server.events_polling.wait_seconds <= 60):
        problems.append("server.events_polling.wait_seconds must be within [0, 60]")
    if server.incoming_polling.interval_seconds <= 0:
        problems.append("server.incoming_polling.interval_seconds must be > 0")
    if server.incoming_polling.offline_delete_threshold < 1:
        problems.append("server.incoming_polling.offline_delete_threshold must be >= 1")
    if server.incoming_polling.inconclusive_delete_threshold < 0:
        problems.append("server.incoming_polling.inconclusive_delete_threshold must be >= 0 (0 disables it)")
    if server.incoming_polling.inconclusive_min_span_seconds < 0:
        problems.append("server.incoming_polling.inconclusive_min_span_seconds must be >= 0 (0 = count only)")
    if server.cookies.degraded_probe_interval_seconds <= 0:
        problems.append("server.cookies.degraded_probe_interval_seconds must be > 0")
    if server.use_dash_for_youtube and server.cookies.enabled:
        problems.append(
            "use_dash_for_youtube must be turned off when cookies are enabled: a "
            "--live-from-start backfill is a long-lived authenticated download "
            "burst, exactly the pattern that gets accounts flagged"
        )

    if not config.active_streamers:
        problems.append("no active streamers configured")
    seen_keys: set[str] = set()
    for s in config.streamers:
        if not valid_stream_id(s.key):
            problems.append(f"streamer key {s.key!r} must match ^[A-Za-z0-9_-]{{1,128}}$")
        if s.key in seen_keys:
            problems.append(f"duplicate streamer key {s.key!r}")
        seen_keys.add(s.key)
        if s.active and not s.urls and not server.incoming_polling.enabled:
            problems.append(f"streamer {s.key!r} has no urls and incoming_polling is disabled")

    tr = config.transcription
    providers_in_use = [tr.provider] + ([tr.fallback_provider] if tr.fallback_provider else [])
    for name in providers_in_use:
        cls = get_provider_class(name)
        if cls is None:
            problems.append(f"transcription provider must be one of {registered_names()}, got {name!r}")
            continue
        problems.extend(f"transcription.{name}: {issue}" for issue in cls.validate(tr.options_for(name)))
    if tr.fallback_provider and tr.fallback_provider == tr.provider:
        problems.append("transcription.fallback_provider must differ from provider")
    for name in tr.provider_options:
        if get_provider_class(name) is None:
            problems.append(f"transcription.{name}: no such provider (registered: {registered_names()})")
    if tr.concurrency < 1:
        problems.append("transcription.concurrency must be >= 1")
    if not tr.language:
        problems.append("transcription.language must be non-empty")

    if config.capture.probe_concurrency < 1:
        problems.append("capture.probe_concurrency must be >= 1")

    if problems:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(problems))


def check_executables(config: Config) -> list[str]:
    """Return human-readable problems with required external binaries."""
    problems = []
    for name, path in (("yt-dlp", config.capture.yt_dlp_path), ("ffmpeg", config.capture.ffmpeg_path)):
        resolved = shutil.which(path) or (path if Path(path).is_file() and os.access(path, os.X_OK) else None)
        if not resolved:
            problems.append(f"{name} not found or not executable at {path!r}")
    if config.server.cookies.enabled:
        for which in ("check", "download"):
            f = config.cookies_file(which)
            if f is not None and not f.is_file():
                problems.append(f"cookies enabled but {which} file missing: {f}")
    return problems
