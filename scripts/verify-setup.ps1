# Verify the local setup: required tools, dependencies, and config.
#
# Usage: .\scripts\verify-setup.ps1 [config-name.yaml]

param([string]$ConfigName)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[FAIL] uv is not installed - see https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}
Write-Host "  [ok]   uv: $(uv --version)"

Write-Host "Installing dependencies (uv sync --locked)..."
uv sync --locked --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($ConfigName) {
    uv run python scripts/verify_setup.py $ConfigName
} else {
    uv run python scripts/verify_setup.py
}
exit $LASTEXITCODE
