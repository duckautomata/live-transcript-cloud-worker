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
- `streamers[].use_youtube_timeline`, VOD-accurate YouTube capture from the
  live edge (see [Capture](#capture)).
- `transcription.concurrency`, per-channel in-flight provider calls. Results
  are re-ordered before submission, so this affects throughput only, never
  line ordering.

## Capture

| Source | Strategy | Timestamps |
| --- | --- | --- |
| default (Twitch, everything else) | live-edge segmenting | estimated (`vodAccurate: false`) |
| YouTube, on a streamer with `use_youtube_timeline: true` | YouTube timeline | exact (`vodAccurate: true`) |

**Live-edge** pipes `yt-dlp -o -` into ffmpeg's segment muxer (`-c copy -f
segment -segment_format mpegts`), producing independently decodable,
byte-concatenable chunks, the format the server's clip pipeline requires.
Timestamps are estimated from each segment's own write time and self-heal
after stalls. A stall watchdog (`stale_threshold.ytdlp_seconds`) terminates a
wedged yt-dlp, and a no-audio watchdog (`stale_threshold.no_audio_seconds`)
reconnects when segments keep arriving but none carry usable audio.

**YouTube timeline** (`streamers[].use_youtube_timeline: true`) joins at the
live edge, like live-edge capture, but every line is VOD-accurate, without
downloading the stream from the start.

YouTube stamps every live segment with its own media clock (the PTS of a video
segment, or the ID3 timestamp at the start of an audio segment), and that
clock is the VOD timeline. The worker reads the video and audio HLS playlists
that yt-dlp resolves, downloads each segment itself, pairs video and audio by
sequence number, and stamps each line with `stream start time + clock of the
first audio sample` (plus 36 ms, the measured constant between YouTube's HLS
clock and its VOD). Video and audio are muxed with ffmpeg `-copyts`, so both
keep the clock and stay in sync. Chunks are MPEG-TS of about
`server.buffer_size_seconds` (closed at the segment boundary nearest it), like
live-edge ones, so media upload and clips work the same.

When enabled, it is the only strategy used for that streamer's YouTube URLs:
it never falls back to live-edge capture, so no line is ever approximate.
Twitch URLs are unaffected.

How a run behaves:

- **It stays up while the stream is live.** A frozen playlist, rejected URLs
  or unusable data (nothing usable for `stale_threshold.no_audio_seconds`)
  rebuild the connection: yt-dlp re-resolves the stream, and its runs are
  paced and back off.
- **It ends only when yt-dlp says the stream isn't live** (or can't be
  watched), when the stream's playlist ends, or when the worker is stopped or
  restarted. The watcher then re-probes as usual.
- **Expiring URLs are refreshed early** without dropping the current
  connection.
- **A restart picks up where the last run stopped**
  (`tmp/{key}/youtube_timeline_state.json`), catching up from up to
  `stale_threshold.lfs_gap_seconds` behind live. While yt-dlp keeps failing it
  waits at most half of that between attempts, so a stream coming back from an
  outage is never skipped.
- **A backed-up pipeline doesn't make it skip.** While transcription or the
  server falls behind, the pipeline's bounded queue holds the capture back,
  and the time it waits widens that catch-up window, so it catches up
  afterwards instead of skipping to the live edge. YouTube's own limit still
  applies: once older segments stop being served (three 404s in a row, which
  includes the stretch before a stream break or before YouTube changed its
  segment URLs), it continues near the live edge, as after any outage.
- **Outages don't cost VOD accuracy.** YouTube's clock, like the VOD, skips
  the time a stream was down, so lines after an outage still land where the
  VOD has them. The worker keeps trusting a clock that continues the timeline
  it already trusted, however long the outages add up to.
- **A video segment that doesn't arrive leaves a gap in the video only.**
  Audio is never held up.
- **A line is `vodAccurate: false` only when the clock can't back it.** If
  YouTube restarts a stream's timeline (seen on 24/7 streams), its clock no
  longer counts from the stream's start, so those lines are stamped by arrival
  (`now - duration - capture.live_latency_seconds`), like live-edge. If the
  probe found no start time for the stream, the server's `startTime` is the
  activation time rather than the clock's zero: lines still follow the stream
  clock, but none claims VOD accuracy (one warning says so).

_Pros_

- VOD-accurate timestamps from the live edge. Works when the streamer has DVR
  disabled.
- Missing part of the stream (an outage, a reconnect) loses only those lines;
  later lines still carry their own clock, so nothing drifts.
- Audio and video stay in sync.
- Works with cookies enabled: its authenticated traffic is one paced
  `yt-dlp -j` per connection, and the segments themselves come from signed
  URLs.

_Cons_

- YouTube only, and only while YouTube serves HLS for live streams. The
  transport is isolated in `youtube_timeline_source.py` so a SABR source can
  be added when yt-dlp supports it.
- Starts at the live edge: whatever happened before the worker joined is not
  transcribed.

Twitch always captures at the live edge (the Twitch live-from-start path from
the reference worker is intentionally not implemented).

> **DASH capture was removed.** The YouTube timeline replaces it: the same
> exact timestamps, without downloading the stream from the start. A config
> that still sets `server.use_dash_for_youtube` or
> `server.stale_threshold.fragment_seconds` is rejected at startup with a
> pointer to `use_youtube_timeline`, and the old capture's leftovers
> (`tmp/{key}/fragments/`, `tmp/{key}/dash_state.json`) are deleted at boot.

### Diagnostics files

The worker's own log is `tmp/_logs/app.log` (JSON lines, rotated at 5 MB).
The output of the external tools is not in it; it goes to per-channel files
under `tmp/{key}/`, each run stamped with a header line, so when a capture
ends without producing anything the reason is on disk:

- `ytdlp.log`: live-edge capture. The exact yt-dlp command line and the
  stderr of yt-dlp and ffmpeg.
- `youtube_timeline.log`: YouTube timeline. yt-dlp's output for each URL
  resolution, the formats used, reconnects and their reasons, gaps in the
  stream, failed segment downloads or muxes, and a footer with the run's
  outcome.
- `stream_stats.log`: the raw `yt-dlp -j` output of stream probes, useful
  when a stream is not detected as live.

These files are trimmed to their most recent ~256 KB once they grow past
512 KB.

## Reliability behaviour

- Retry classification per the spec: network/5xx/429 retry with exponential
  backoff + jitter (`Retry-After` honoured); 400/403 never retried; 404 retried
  only for media uploads; `409` on a line triggers a full `/sync` resync.
- Per-channel durable state: append-only `transcript.jsonl` (crash-torn tails
  are truncated on load), atomic `meta.json`, and a disk-backed media queue,
  a worker restart mid-stream resumes the line sequence without a resync. The
  YouTube timeline also keeps an atomic resume point
  (`youtube_timeline_state.json`), so a restart continues from the next
  segment instead of replaying lines.
- Media uploads run in a channel-fair pool, never block line submission, and
  survive restarts (the queue directory is rescanned at boot).
- Live-edge capture has a stall watchdog (`stale_threshold.ytdlp_seconds`);
  the YouTube timeline rebuilds a connection that stops advancing and paces
  its yt-dlp runs. The heartbeat runs independently of everything so a
  stalled download can't make the worker look offline.
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
config validation, real-ffmpeg audio extraction, end-to-end capture
(real ffmpeg segmenting driven by a scripted fake yt-dlp, including the
stall watchdog and stop-flush paths), watcher strategy routing, and the
YouTube timeline: segment parsing and clock unwrapping, chunk assembly,
real-ffmpeg muxing, the HLS source against a mock HTTP transport, and the
run loop's reconnect, resume and clock-trust decisions on a fake clock.
