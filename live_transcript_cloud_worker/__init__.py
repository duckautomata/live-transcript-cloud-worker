"""Cloud-native live-transcript worker: yt-dlp capture + remote transcription.

The released version is stamped into the image at build time (Dockerfile
``ARG APP_VERSION``, set from the git tag by the deploy workflow), so it is
read from the environment instead of being hardcoded here: the source tree
is never bumped, and any constant baked in would go stale immediately.
Everything that reports a version, the startup banner and the heartbeat,
must use these helpers so they can never disagree.
"""

from __future__ import annotations

import os

__all__ = ["app_version", "build_time"]

DEV_VERSION = "local"  # a source checkout: no image, no tag
UNKNOWN_BUILD_TIME = "unknown"


def app_version() -> str:
    return os.environ.get("APP_VERSION") or DEV_VERSION


def build_time() -> str:
    return os.environ.get("BUILD_DATE") or UNKNOWN_BUILD_TIME
