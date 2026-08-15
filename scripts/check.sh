#!/usr/bin/env bash

# Run all code quality checks (same gates as CI) - mainly used for development.
#
# Usage: ./scripts/check.sh [options]
# Options:
#   --fix: Auto-fix ruff lint violations where possible

cd "$(dirname "$0")/.."
echo "Running from project root: $PWD"
set -e

LINT_FIX=""
for arg in "$@"; do
    if [ "$arg" == "--fix" ]; then
        LINT_FIX="--fix"
    fi
done

echo -e "\nRunning Ruff formatter..."
uv run ruff format .

echo -e "\nRunning Ruff linter..."
uv run ruff check . $LINT_FIX

echo -e "\nRunning Pyrefly type checker..."
uv run pyrefly check

echo -e "\nAll checks passed!"
