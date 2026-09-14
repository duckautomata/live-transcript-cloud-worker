# live-transcript-cloud-worker

A cloud-friendly version of `live-transcript-worker`: it watches stream URLs,
downloads live streams with **yt-dlp**, cuts them into fixed-duration MPEG-TS
chunks with **ffmpeg**, transcribes each chunk with a **remote speech-to-text
API** (no GPU needed), and pushes every chunk as a numbered transcript line to
`live-transcript-server`.

Because transcription is delegated to a third party, this worker runs on any
small VM or container host, the only local work is stream download, stream
copy/segmenting (`-c copy`, no transcoding), and one cheap audio decode per
chunk.

## Transcription providers

Configurable via `transcription.provider`, with an optional automatic
`fallback_provider` used when the primary exhausts its retries:

| Provider | Model | Price (2026) | Notes |
| --- | --- | --- | --- |
| `deepgram` | nova-3 (pre-recorded) | ~$0.0043/audio-min | raw WAV body, word timestamps, `keyterms` boosting |
| `cloudflare` | `@cf/openai/whisper-large-v3-turbo` | ~$0.0005/audio-min | Workers AI REST, base64 payload, whisper segments |

Credentials come from the config file or environment variables (env wins):

```
LT_SERVER_API_KEY       server.apiKey
DEEPGRAM_API_KEY        transcription.deepgram.api_key
CLOUDFLARE_API_TOKEN    transcription.cloudflare.api_token   (Workers AI template)
CLOUDFLARE_ACCOUNT_ID   transcription.cloudflare.account_id
```

Every provider receives the chunk audio as 16 kHz mono WAV extracted by ffmpeg,
the same decode pass that measures the chunk's exact duration, which drives the
timestamp math. Silence legitimately produces an empty line (line IDs must stay
gapless). If every provider fails on a chunk, the line is still emitted with no
segments so the sequence never gaps.

### Adding a new provider

Providers are plugins in
[`live_transcript_cloud_worker/providers/`](live_transcript_cloud_worker/providers/base.py)

one self-contained module each:

1. Create `providers/<name>.py` with a `TranscriptionProvider` subclass
   decorated with `@register`: declare `name`, the YAML option keys it accepts
   (`known_options`), any env-var credential overrides (`env_overrides`),
   config validation, and one `async transcribe(wav) -> [(start, text), ...]`
   method that raises `TranscriptionError(msg, retryable=...)` on failure.
2. Import the module in `providers/__init__.py`.
3. Point `transcription.provider: <name>` at it and add a
   `transcription.<name>:` options block in the YAML.

Option parsing, config validation, env overrides, retry/backoff, and fallback
chaining all come from the framework, the module only speaks the vendor's
HTTP API. `providers/base.py`'s docstring has a full skeleton.

## Setup

Requires:
- Python ≥3.12
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg`
- `yt-dlp`


(binary paths configurable under `capture:`).

```bash
uv sync
cp config/example.yaml config/config.yaml   # then edit
uv run main.py                              # or: uv run main.py dev.yaml
```

`config/example.yaml` documents every knob. Highlights:

- `server.enabled: false`, local mode: no server calls; transcripts append to
  `tmp/{key}/transcript.text`. The easiest way to validate provider credentials
  and capture on your machine.
- `server.cookies`, passes `--cookies <file>` (Netscape format, project root)
  to yt-dlp. **Needed in datacenters**, where YouTube blocks anonymous
  downloads; leave disabled when running from a residential IP. Twitch never
  uses cookies. `check_filename` / `download_filename` let you rotate the
  polling and download identities independently.
- `streamers[].media_type`, `none` (transcript only), `audio`, or `video`;
  chunks are uploaded to the server after their line and become client-side
  playback/clips.
- `transcription.concurrency`, per-channel in-flight provider calls. Results
  are re-ordered before submission, so this affects throughput only, never
  line ordering.

## Capture

| Source | Strategy | Timestamps |
| --- | --- | --- |
| default (Twitch, everything else) | live-edge segmenting | estimated (`vodAccurate: false`) |
| YouTube with `use_dash_for_youtube: true` | DASH `--live-from-start` | exact (`vodAccurate: true`) |

**Live-edge** pipes `yt-dlp -o -` into ffmpeg's segment muxer (`-c copy -f
segment -segment_format mpegts`), producing independently decodable,
byte-concatenable chunks, the format the server's clip pipeline requires.
Timestamps are estimated from each segment's own write time and self-heal
after stalls. A stall watchdog (`stale_threshold.ytdlp_seconds`) terminates a
wedged yt-dlp.

**YouTube DASH** (`use_dash_for_youtube: true`) downloads the stream's
fragments from second zero, merges each sequence with ffmpeg, and emits
~6 s chunks with exact VOD timestamps. It carries the reference worker's full
guardrail set: atomic resume state (a worker restart continues instead of
replaying duplicate lines), Frag1 byte-compare continuity verification with
sequence-reset recovery, stale-fragment force-through
(`stale_threshold.fragment_seconds`), a stall watchdog, and automatic
fallback to live-edge when yt-dlp fails before producing fragments or capture
falls more than `stale_threshold.lfs_gap_seconds` behind live (checked
pre-flight too, so a worker that was down for an hour doesn't try to catch up
from an hour back).

> **DASH must be turned off when cookies are enabled.** A `--live-from-start`
> backfill is a long-lived authenticated download burst, exactly the pattern
> that gets accounts flagged. The config refuses to start with both enabled;
> pick cookies (datacenter) or DASH (residential IP), not both.

Twitch always captures at the live edge regardless of this setting (the
Twitch live-from-start path from the reference worker is intentionally not
implemented).

## Reliability behaviour

- Retry classification per the spec: network/5xx/429 retry with exponential
  backoff + jitter (`Retry-After` honoured); 400/403 never retried; 404 retried
  only for media uploads; `409` on a line triggers a full `/sync` resync.
- Per-channel durable state: append-only `transcript.jsonl` (crash-torn tails
  are truncated on load), atomic `meta.json`, and a disk-backed media queue,
  a worker restart mid-stream resumes the line sequence without a resync.
- Media uploads run in a channel-fair pool, never block line submission, and
  survive restarts (the queue directory is rescanned at boot).
- Capture has a stall watchdog (`stale_threshold.ytdlp_seconds`), and the
  heartbeat runs independently of everything so a stalled download can't make
  the worker look offline.
- Incoming-queue hygiene: a queued URL is removed once its stream is confirmed
  over, and a URL whose probes cannot determine liveness at all (private or
  removed video, bot check, members-only entry, unknown status) is given up on
  after `incoming_polling.inconclusive_delete_threshold` consecutive probes
  spanning at least `incoming_polling.inconclusive_min_span_seconds`; probe
  timeouts, yt-dlp network/rate-limit/extractor failures and other worker-side
  errors never touch the queue. The worker re-reads the queue before a probe
  when its copy is older than `incoming_polling.interval_seconds`, so a URL
  removed on the admin page stops being probed within about one probe
  interval.
- A dead YouTube cookie jar (yt-dlp reports it is scraping anonymously) parks
  YouTube URLs on `cookies.degraded_probe_interval_seconds` instead of acting
  on verdicts it cannot trust. Overwrite the existing `cookies.txt` in place
  (`cat new.txt > cookies.txt`, `cp new.txt cookies.txt`, or scp onto the
  existing path) and the parked URLs are re-probed at once without a restart.
  Never replace it by rename (`mv`, `rsync` without `--inplace`, an editor's
  atomic save): the deployed stack bind-mounts the single file into the
  container, so a rename leaves the container and yt-dlp on the old inode and
  the worker cannot see the new jar (no "cookies.txt changed" log line
  appears). If that has already happened, restart the worker container. One
  more yt-dlp habit to know: it writes its in-memory jar back when it exits,
  so a capture that was already running when you overwrote a shared
  `cookies.txt` puts the old cookies back when it ends. Point
  `check_filename` and `download_filename` at different files, or copy the
  jar again after the capture.
- Graceful shutdown on SIGINT/SIGTERM: capture unwinds, the transcription
  backlog drains within a bounded budget, pending uploads get a final window,
  every live channel is deactivated.

## Running in Docker

```bash
docker compose up -d
```

The provided `Dockerfile` installs ffmpeg, deno (needed by current yt-dlp for
YouTube), and a pinned yt-dlp release. `tmp/` is a volume so per-channel state
and queued media survive restarts. Provide secrets via environment variables in
`docker-compose.yml` rather than baking them into the config.

To test an image without going through the GitHub deploy pipeline:

```bash
./scripts/local-build.sh
```

builds and pushes `duckautomata/live-transcript-cloud-worker:dev` from your
machine (resolving the latest yt-dlp/deno releases, same as CI). Use
`--no-push` to only build for local `docker run` testing, or
`NEW_VERSION=x.y.z ./scripts/local-build.sh` to cut an emergency
versioned+latest release by hand. Pushing requires a prior `docker login`.

## Scripts

Each script has a bash version (Linux/macOS) and a PowerShell version
(Windows) with identical behavior:

| Script | Purpose |
| --- | --- |
| `scripts/verify-setup.sh` / `.ps1` | Verify the local setup: uv, dependencies (`uv sync --locked`), python version, ffmpeg/yt-dlp/deno/docker, and config validation (`[config-name.yaml]` arg, default `config.yaml`; falls back to checking that `example.yaml` parses). Fails loudly on anything required. |
| `scripts/check.sh` / `.ps1` | The CI quality gates: `ruff format`, `ruff check` (`--fix` / `-Fix` to auto-fix), `pyrefly check`. |
| `scripts/local-build.sh` / `.ps1` | Build + push the Docker image without going through GitHub (see "Running in Docker"). |

## Tests

```bash
uv run pytest
```

Contract tests cover the server-facing invariants (gapless IDs, 409→sync,
retry classification, events cursor echoing, restart ack semantics), state
crash-recovery, provider response parsing/retryability and the plugin
registry, pipeline ordering, probe classification and watcher scheduling,
config validation, real-ffmpeg audio extraction, and end-to-end capture
(real ffmpeg segmenting driven by a scripted fake yt-dlp, including the
stall watchdog and stop-flush paths).
