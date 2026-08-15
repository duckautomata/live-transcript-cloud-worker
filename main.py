"""Entry point.

Usage: uv run main.py [config-name.yaml]

The argument names a file under config/ (default config.yaml), matching the
reference worker's invocation shape so existing run scripts carry over.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from live_transcript_cloud_worker.app import run_app
from live_transcript_cloud_worker.config import ConfigError, load_config
from live_transcript_cloud_worker.logging_setup import setup_logging


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    setup_logging(base_dir)
    try:
        config = load_config(base_dir / "config" / config_name, base_dir)
    except ConfigError as exc:
        logging.getLogger(__name__).critical("%s", exc)
        return 1
    try:
        return asyncio.run(run_app(config))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
