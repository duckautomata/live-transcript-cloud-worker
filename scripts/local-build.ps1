# Build and push the worker image from this machine - the fast path for
# testing an image without going through the GitHub deploy pipeline.
#
# Usage:
#   .\scripts\local-build.ps1              build + push :dev
#   .\scripts\local-build.ps1 -NoPush      build only (local docker run testing)
#   $env:NEW_VERSION="1.2.3"; .\scripts\local-build.ps1
#                                          build + push :1.2.3 and :latest
#
# Requires: docker (logged in to Docker Hub for pushes: `docker login`).

param([switch]$NoPush)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$ImageName = "duckautomata/live-transcript-cloud-worker"
$BuildDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

# Same resolution the deploy workflows use: latest release TAGS, passed to
# the Dockerfile verbatim (yt-dlp tags have no "v", deno tags include it).
function Resolve-LatestTag([string]$Repo) {
    (Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest").tag_name
}

$YtdlpVersion = Resolve-LatestTag "yt-dlp/yt-dlp"
$DenoVersion = Resolve-LatestTag "denoland/deno"
Write-Host "Resolved yt-dlp version: $YtdlpVersion"
Write-Host "Resolved Deno version:   $DenoVersion"

if (-not $env:NEW_VERSION) {
    $AppVersion = "dev"
    $Tags = @("${ImageName}:dev")
    Write-Host "No NEW_VERSION provided; building the dev tag..."
} else {
    $AppVersion = $env:NEW_VERSION
    $Tags = @("${ImageName}:$($env:NEW_VERSION)", "${ImageName}:latest")
    Write-Host "Building version $($env:NEW_VERSION) (tags: $($Tags -join ', '))..."
}

$TagArgs = @()
foreach ($Tag in $Tags) { $TagArgs += @("-t", $Tag) }

docker build `
    --build-arg APP_VERSION=$AppVersion `
    --build-arg BUILD_DATE=$BuildDate `
    --build-arg YTDLP_VERSION=$YtdlpVersion `
    --build-arg DENO_VERSION=$DenoVersion `
    @TagArgs `
    .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($NoPush) {
    Write-Host "Done (build only). Built: $($Tags -join ', ')"
    exit 0
}

foreach ($Tag in $Tags) {
    Write-Host "Pushing $Tag..."
    docker push $Tag
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "Done. Published: $($Tags -join ', ')"
