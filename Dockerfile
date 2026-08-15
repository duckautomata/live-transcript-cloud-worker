FROM python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_FROZEN=1 \
    UV_NO_SYNC=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /usr/local/bin/

# Version args take the release TAG verbatim (deno tags include the "v",
# yt-dlp tags don't), the deploy workflows resolve the latest tags and pass
# them through. Both downloads are checksum-verified.
ARG DENO_VERSION=v2.4.2
RUN cd /tmp \
    && curl -fsSL --retry 5 --retry-delay 2 "https://github.com/denoland/deno/releases/download/${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip" \
         -o deno-x86_64-unknown-linux-gnu.zip \
    && curl -fsSL --retry 5 --retry-delay 2 "https://github.com/denoland/deno/releases/download/${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip.sha256sum" \
         -o deno-x86_64-unknown-linux-gnu.zip.sha256sum \
    && sha256sum -c deno-x86_64-unknown-linux-gnu.zip.sha256sum \
    && unzip -q deno-x86_64-unknown-linux-gnu.zip -d /usr/local/bin \
    && rm deno-x86_64-unknown-linux-gnu.zip deno-x86_64-unknown-linux-gnu.zip.sha256sum \
    && deno --version

ARG YTDLP_VERSION=2026.08.05
RUN cd /tmp \
    && curl -fsSL --retry 5 --retry-delay 2 "https://github.com/yt-dlp/yt-dlp/releases/download/${YTDLP_VERSION}/yt-dlp" -o yt-dlp \
    && curl -fsSL --retry 5 --retry-delay 2 "https://github.com/yt-dlp/yt-dlp/releases/download/${YTDLP_VERSION}/SHA2-256SUMS" -o SHA2-256SUMS \
    && grep '  yt-dlp$' SHA2-256SUMS | sha256sum -c - \
    && rm SHA2-256SUMS \
    && install -m 755 yt-dlp /usr/local/bin/yt-dlp \
    && rm yt-dlp \
    && yt-dlp --version

WORKDIR /app
RUN mkdir -p /app/tmp
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY main.py ./
COPY live_transcript_cloud_worker/ ./live_transcript_cloud_worker/

ARG APP_VERSION=local
ARG BUILD_DATE=unknown
ENV APP_VERSION=${APP_VERSION} \
    BUILD_DATE=${BUILD_DATE}

VOLUME ["/app/tmp"]
CMD ["uv", "run", "--no-dev", "main.py"]
