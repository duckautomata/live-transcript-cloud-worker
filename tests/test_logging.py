"""JSON log formatter: shape must stay promtail/Grafana-compatible and
mirror the live-transcript-server's slog output."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

from live_transcript_cloud_worker.logging_setup import JsonFormatter

RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def make_record(msg, *args, level=logging.INFO, exc_info=None, extra=None):
    logger = logging.getLogger("live_transcript_cloud_worker.testmod")
    record = logger.makeRecord(logger.name, level, "file.py", 1, msg, args, exc_info, func="do_thing", extra=extra)
    return record


def fmt(record) -> dict:
    return json.loads(JsonFormatter().format(record))


def test_core_fields_and_rfc3339_time():
    entry = fmt(make_record("hello %s", "world"))
    assert entry["msg"] == "hello world"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "live_transcript_cloud_worker.testmod"
    assert entry["func"] == "do_thing"
    assert RFC3339_RE.match(entry["time"]), entry["time"]
    # Round-trips as an aware UTC datetime (what promtail's RFC3339 parse sees).
    parsed = datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    assert abs(parsed.timestamp() - datetime.now(UTC).timestamp()) < 5


def test_level_names_match_slog():
    assert fmt(make_record("x", level=logging.WARNING))["level"] == "WARN"
    assert fmt(make_record("x", level=logging.CRITICAL))["level"] == "ERROR"
    assert fmt(make_record("x", level=logging.DEBUG))["level"] == "DEBUG"


def test_channel_key_prefix_lifted_into_field():
    entry = fmt(make_record("[doki] line %d posted", 7))
    assert entry["key"] == "doki"
    assert entry["msg"] == "line 7 posted"
    # Non-key-shaped prefixes stay in the message untouched.
    entry = fmt(make_record("plain message"))
    assert "key" not in entry
    assert entry["msg"] == "plain message"


def test_extra_fields_included():
    entry = fmt(make_record("uploaded", extra={"streamId": "abc", "elapsedMs": 412}))
    assert entry["streamId"] == "abc"
    assert entry["elapsedMs"] == 412


def test_exception_lands_in_err():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        entry = fmt(make_record("it failed", level=logging.ERROR, exc_info=sys.exc_info()))
    assert "ValueError: boom" in entry["err"]
    assert entry["msg"] == "it failed"


def test_output_is_one_json_line():
    text = JsonFormatter().format(make_record("[doki] multi\nline"))
    assert "\n" not in text.replace("\\n", "")
    json.loads(text)  # valid JSON


def test_field_order_keeps_msg_before_metadata():
    """The Grafana logs dashboard splits each line at the "msg" field with
    a regex (everything after it becomes the indented detail line), so
    time/level/msg must stay a prefix, with logger/func/key/extras after.
    """
    line = JsonFormatter().format(make_record("[doki] hello", extra={"n": 1}))
    assert re.match(r'^\{"time":"[^"]+","level":"INFO","msg":', line)
    rest = re.match(r'^\{.*?"msg":".*?"(?:,(?P<rest>.*))?\}$', line).group("rest")
    assert '"logger":' in rest and '"key":"doki"' in rest and '"n":1' in rest
