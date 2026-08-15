#!/usr/bin/env bash

# Verify the local setup: required tools, dependencies, and config.
#
# Usage: ./scripts/verify-setup.sh [config-name.yaml]

cd "$(dirname "$0")/.."
set -e

if ! command -v uv &>/dev/null; then
    echo "[FAIL] uv is not installed - see https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
echo "  [ok]   uv: $(uv --version)"

echo "Installing dependencies (uv sync --locked)..."
uv sync --locked --quiet

uv run python scripts/verify_setup.py "$@"
