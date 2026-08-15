"""Verify the local setup: tools, dependencies, and config.

Run through the platform wrappers (scripts/verify-setup.sh or
scripts/verify-setup.ps1), which install dependencies first, or directly:

    uv run python scripts/verify_setup.py [config-name.yaml]

Exits non-zero when anything required is broken. Warnings ([warn]) don't
fail the check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures = 0


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def fail(msg: str) -> None:
    global failures
    failures += 1
    print(f"  [FAIL] {msg}")


def tool_version(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0]


def check_python() -> None:
    if sys.version_info >= (3, 12):  # noqa: UP036, guards direct system-python runs
        ok(f"python {sys.version.split()[0]}")
    else:
        fail(f"python >= 3.12 required, running {sys.version.split()[0]}")


def check_tools(yt_dlp_path: str, ffmpeg_path: str) -> None:
    for name, path, required in (
        ("ffmpeg", ffmpeg_path, True),
        ("yt-dlp", yt_dlp_path, True),
        ("deno", "deno", False),  # current yt-dlp needs it for YouTube
        ("docker", "docker", False),  # only for scripts/local-build
    ):
        resolved = shutil.which(path) or (path if Path(path).is_file() else None)
        if resolved is None:
            if required:
                fail(f"{name} not found (looked for {path!r})")
            else:
                warn(
                    f"{name} not found, "
                    + ("YouTube capture may fail without it" if name == "deno" else "needed only for scripts/local-build")
                )
            continue
        flag = "-version" if name == "ffmpeg" else "--version"
        version = tool_version([resolved, flag])
        ok(f"{name}: {version or resolved}")


def check_config() -> tuple[str, str]:
    """Validate the config; returns (yt_dlp_path, ffmpeg_path) to probe."""
    from live_transcript_cloud_worker.config import ConfigError, check_executables, load_config

    name = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    path = ROOT / "config" / name
    if not path.is_file():
        warn(f"config/{name} not found, copy config/example.yaml to config/config.yaml and edit it")
        # Still prove the example parses so code + example never drift.
        os.environ.setdefault("DEEPGRAM_API_KEY", "verify-setup-placeholder")
        os.environ.setdefault("LT_SERVER_API_KEY", "verify-setup-placeholder")
        try:
            config = load_config(ROOT / "config" / "example.yaml", ROOT)
            ok("config/example.yaml parses")
        except ConfigError as exc:
            fail(f"config/example.yaml is invalid: {exc}")
            return "yt-dlp", "ffmpeg"
    else:
        try:
            config = load_config(path, ROOT)
            ok(f"config/{name} is valid ({len(config.active_streamers)} active channel(s), provider={config.transcription.provider})")
        except ConfigError as exc:
            fail(f"config/{name} is invalid: {exc}")
            return "yt-dlp", "ffmpeg"
        for problem in check_executables(config):
            fail(problem)
        if config.server.enabled:
            ok(f"server mode: {config.server.url}")
        else:
            warn("server.enabled: false, local mode (transcripts go to tmp/{key}/transcript.text)")
    return config.capture.yt_dlp_path, config.capture.ffmpeg_path


def main() -> int:
    print(f"Verifying setup in {ROOT}")
    check_python()
    yt_dlp_path, ffmpeg_path = check_config()
    check_tools(yt_dlp_path, ffmpeg_path)
    if failures:
        print(f"\n{failures} problem(s) found.")
        return 1
    print("\nSetup looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
