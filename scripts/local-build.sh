#!/usr/bin/env bash
# Build and push the worker image from this machine, the fast path for
# testing an image without going through the GitHub deploy pipeline.
#
# Usage:
#   ./scripts/local-build.sh              build + push duckautomata/live-transcript-cloud-worker:dev
#   ./scripts/local-build.sh --no-push    build only (local docker run / compose testing)
#   NEW_VERSION=1.2.3 ./scripts/local-build.sh
#                                         build + push :1.2.3 and :latest (normally the
#                                         deploy workflow's job, use for emergencies)
#
# Requires: docker (logged in to Docker Hub for pushes: `docker login`).

set -euo pipefail
cd "$(dirname "$0")/.."

PUSH=1
if [ "${1:-}" = "--no-push" ]; then
    PUSH=0
fi

IMAGE_NAME="duckautomata/live-transcript-cloud-worker"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Same resolution the deploy workflows use: latest release TAGS, passed to the
# Dockerfile verbatim (yt-dlp tags have no "v", deno tags include it).
resolve_version() {
    curl -fsSI "https://github.com/$1/releases/latest" \
        | grep -i '^location:' \
        | sed 's|.*/tag/||' \
        | tr -d '[:space:]'
}

YTDLP_VERSION=$(resolve_version "yt-dlp/yt-dlp")
DENO_VERSION=$(resolve_version "denoland/deno")
echo "Resolved yt-dlp version: ${YTDLP_VERSION}"
echo "Resolved Deno version:   ${DENO_VERSION}"

if [ -z "${NEW_VERSION:-}" ]; then
    APP_VERSION="dev"
    TAGS=("${IMAGE_NAME}:dev")
    echo "No NEW_VERSION provided; building the dev tag..."
else
    APP_VERSION="${NEW_VERSION}"
    TAGS=("${IMAGE_NAME}:${NEW_VERSION}" "${IMAGE_NAME}:latest")
    echo "Building version ${NEW_VERSION} (tags: ${TAGS[*]})..."
fi

TAG_ARGS=()
for tag in "${TAGS[@]}"; do
    TAG_ARGS+=(-t "$tag")
done

docker build \
    --build-arg APP_VERSION="${APP_VERSION}" \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg YTDLP_VERSION="${YTDLP_VERSION}" \
    --build-arg DENO_VERSION="${DENO_VERSION}" \
    "${TAG_ARGS[@]}" \
    .

if [ "$PUSH" -eq 0 ]; then
    echo "Done (build only). Built: ${TAGS[*]}"
    exit 0
fi

for tag in "${TAGS[@]}"; do
    echo "Pushing ${tag}..."
    docker push "$tag"
done
echo "Done. Published: ${TAGS[*]}"
