# Run all code quality checks (same gates as CI) - mainly used for development.
#
# Usage: .\scripts\check.ps1 [-Fix]

param([switch]$Fix)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
Write-Host "Running from project root: $(Get-Location)"

Write-Host "`nRunning Ruff formatter..."
uv run ruff format .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nRunning Ruff linter..."
if ($Fix) { uv run ruff check . --fix } else { uv run ruff check . }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nRunning Pyrefly type checker..."
uv run pyrefly check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nAll checks passed!"
