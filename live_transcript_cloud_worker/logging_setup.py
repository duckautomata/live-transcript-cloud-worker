"""Logging: human-readable console, JSON-lines rotating file.

The file log (tmp/_logs/app.log) is shaped for promtail/Loki/Grafana and
matches the live-transcript-server's slog output so one pipeline config
serves both:

    {"time":"2026-08-15T00:41:47.882744Z","level":"INFO","msg":"...",
     "logger":"live_transcript_cloud_worker.watcher","func":"run","key":"doki"}

- ``time`` is RFC3339 UTC (Z suffix, microseconds), parseable by promtail's
  ``format: RFC3339`` stage.
- ``level`` uses the slog names: DEBUG/INFO/WARN/ERROR (WARNING -> WARN,
  CRITICAL -> ERROR) so Grafana level labels stay uniform across services.
- Messages written as ``[<channel key>] ...`` get the key lifted into a
  structured ``key`` field, mirroring the server's per-channel field.
- ``extra={...}`` fields on a log call are included verbatim; exceptions
  land in ``err``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
from datetime import UTC, datetime
from pathlib import Path

CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_LEVEL_NAMES = {"WARNING": "WARN", "CRITICAL": "ERROR"}
_KEY_PREFIX_RE = re.compile(r"^\[([A-Za-z0-9_-]{1,128})\] ")
# Attributes every LogRecord carries; anything else came from extra={...}.
_RESERVED_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        entry = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": _LEVEL_NAMES.get(record.levelname, record.levelname),
            "msg": msg,
            "logger": record.name,
            "func": record.funcName,
        }
        key_match = _KEY_PREFIX_RE.match(msg)
        if key_match:
            entry["key"] = key_match.group(1)
            entry["msg"] = msg[key_match.end() :]
        for name, value in record.__dict__.items():
            if name not in _RESERVED_ATTRS and not name.startswith("_") and name not in entry:
                entry[name] = value
        if record.exc_info:
            entry["err"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False, separators=(",", ":"))


def setup_logging(base_dir: Path, console_level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)

    log_dir = base_dir / "tmp" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=10, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    # Third-party chatter stays out of the DEBUG file.
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
