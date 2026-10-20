"""
Media plumbing for the YouTube Timeline capture (capture_timeline): everything
that touches segment bytes, with no network access.

YouTube stamps every live segment with its own media clock: the PTS of an
MPEG-TS segment, the ID3 PRIV frame (`com.apple.streaming.transportStreamTimestamp`)
at the head of a raw ADTS audio segment, or the tfdt of a fragmented-MP4
segment. That clock is the VOD timeline (measured on HLS: VOD time == clock +
0.036 s, constant across a 3.3 h stream and across a stream break), so a chunk's
VOD position is the clock of its first audio sample. Nothing here derives time
from arrival times, file mtimes or summed durations.

Parsers return None instead of raising on malformed input: a segment that can't
be read is treated as missing by the caller.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# MPEG-TS timestamps are 33-bit values at 90 kHz, so the clock wraps every ~26.5 h.
TS_CLOCK_HZ = 90_000
TS_WRAP_TICKS = 1 << 33
TS_WRAP_SECONDS = TS_WRAP_TICKS / TS_CLOCK_HZ

TS_PACKET_SIZE = 188
TS_SYNC_BYTE = 0x47

ID3_TIMESTAMP_OWNER = b"com.apple.streaming.transportStreamTimestamp\x00"

# Containers a TrackSegment can carry. fmp4 is what a SABR source would deliver.
CONTAINER_MPEGTS = "mpegts"
CONTAINER_ADTS = "adts"
CONTAINER_FMP4 = "fmp4"

# Track names. A muxed track carries video and audio in one MPEG-TS stream.
TRACK_AUDIO = "audio"
TRACK_VIDEO = "video"
TRACK_MUXED = "muxed"

# Continuity tolerance for segments whose duration is only nominal (playlist value).
NOMINAL_TOLERANCE_SECONDS = 0.25

_ADTS_SAMPLE_RATES = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350)
_FMP4_BOX_TYPES = (b"ftyp", b"styp", b"moof", b"sidx", b"emsg", b"moov", b"free", b"prft")


# ----------------------------------------------------------------------
# ID3 / ADTS
# ----------------------------------------------------------------------


def _syncsafe(b: bytes) -> int:
    return (b[0] & 0x7F) << 21 | (b[1] & 0x7F) << 14 | (b[2] & 0x7F) << 7 | (b[3] & 0x7F)


def _id3_tag_length(data: bytes, pos: int) -> int:
    """Total length (header + body + optional footer) of an ID3v2 tag at pos, or 0 if there is none."""
    if len(data) - pos < 10 or data[pos : pos + 3] != b"ID3":
        return 0
    if data[pos + 3] == 0xFF or data[pos + 4] == 0xFF or any(b & 0x80 for b in data[pos + 6 : pos + 10]):
        return 0
    footer = 10 if data[pos + 5] & 0x10 else 0
    return 10 + _syncsafe(data[pos + 6 : pos + 10]) + footer


def parse_id3_timestamp(data: bytes) -> float | None:
    """The segment's clock in seconds from the ID3 tag that heads an HLS raw audio
    segment, or None when the tag or the timestamp frame is absent. The value
    is the raw 33-bit clock (see ClockUnwrapper for the wrap)."""
    tag_len = _id3_tag_length(data, 0)
    if not tag_len:
        return None
    tag = data[:tag_len]
    i = tag.find(ID3_TIMESTAMP_OWNER)
    if i < 0:
        return None
    start = i + len(ID3_TIMESTAMP_OWNER)
    if len(tag) < start + 8:
        return None
    ticks = int.from_bytes(tag[start : start + 8], "big") & (TS_WRAP_TICKS - 1)
    return ticks / TS_CLOCK_HZ


@dataclass(frozen=True)
class AdtsInfo:
    frames: int
    samples: int
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.samples / self.sample_rate


def parse_adts(data: bytes) -> AdtsInfo | None:
    """Counts the complete ADTS frames in a raw AAC segment (ID3 tags anywhere
    between frames are skipped). None when the data holds no valid frame or the
    frames disagree on the sample rate. A truncated last frame is not counted."""
    pos = 0
    frames = 0
    samples = 0
    sample_rate = 0
    n = len(data)
    while pos < n:
        tag_len = _id3_tag_length(data, pos)
        if tag_len:
            pos += tag_len
            continue
        if n - pos < 7:
            break
        b = data[pos : pos + 7]
        if b[0] != 0xFF or (b[1] & 0xF6) != 0xF0:
            # Not a frame boundary: the segment is corrupt from here on.
            break
        sr_index = (b[2] >> 2) & 0x0F
        if sr_index >= len(_ADTS_SAMPLE_RATES):
            break
        frame_len = ((b[3] & 0x03) << 11) | (b[4] << 3) | (b[5] >> 5)
        header_len = 7 if b[1] & 0x01 else 9
        if frame_len <= header_len or pos + frame_len > n:
            break
        rate = _ADTS_SAMPLE_RATES[sr_index]
        if sample_rate and rate != sample_rate:
            return None
        sample_rate = rate
        frames += 1
        samples += 1024 * ((b[6] & 0x03) + 1)
        pos += frame_len
    if not frames:
        return None
    return AdtsInfo(frames=frames, samples=samples, sample_rate=sample_rate)


# ----------------------------------------------------------------------
# MPEG-TS
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TsInfo:
    """Earliest presentation time (raw 33-bit clock, seconds) of the video and
    audio elementary streams found in a TS segment."""

    video_pts: float | None
    audio_pts: float | None


def _ts_sync_offset(data: bytes, start: int = 0) -> int | None:
    """Offset of the next TS packet at or after `start`: a sync byte confirmed by
    the following packet's sync byte (or by the end of the data)."""
    n = len(data)
    i = data.find(bytes([TS_SYNC_BYTE]), start)
    while i >= 0 and i + TS_PACKET_SIZE <= n:
        if i + TS_PACKET_SIZE == n or data[i + TS_PACKET_SIZE] == TS_SYNC_BYTE:
            return i
        i = data.find(bytes([TS_SYNC_BYTE]), i + 1)
    return None


def _earliest(ticks: list[int]) -> int | None:
    """Earliest of a few 33-bit timestamps from one segment, robust to a wrap inside it."""
    if not ticks:
        return None
    ref = ticks[0]
    half = TS_WRAP_TICKS // 2
    deltas = [((t - ref + half) % TS_WRAP_TICKS) - half for t in ticks]
    return (ref + min(deltas)) % TS_WRAP_TICKS


def scan_ts_pts(data: bytes) -> TsInfo | None:
    """Scans the PES headers of a TS segment for PTS values. None when the data
    isn't MPEG-TS or carries no timestamped audio/video PES."""
    pos = _ts_sync_offset(data)
    if pos is None:
        return None
    video: list[int] = []
    audio: list[int] = []
    n = len(data)
    while pos + TS_PACKET_SIZE <= n:
        if data[pos] != TS_SYNC_BYTE:
            # Lost sync: look for the next packet boundary.
            nxt = _ts_sync_offset(data, pos + 1)
            if nxt is None:
                break
            pos = nxt
            continue
        b1, b3 = data[pos + 1], data[pos + 3]
        payload_start = b1 & 0x40
        afc = (b3 >> 4) & 0x03
        off = pos + 4
        if afc & 0x02:
            off += 1 + data[off]
        if payload_start and afc & 0x01 and off + 14 <= pos + TS_PACKET_SIZE:
            p = data[off : pos + TS_PACKET_SIZE]
            if p[0:3] == b"\x00\x00\x01" and p[7] & 0x80:
                stream_id = p[3]
                pts = ((p[9] >> 1) & 0x07) << 30 | p[10] << 22 | (p[11] >> 1) << 15 | p[12] << 7 | p[13] >> 1
                if 0xE0 <= stream_id <= 0xEF:
                    video.append(pts)
                elif 0xC0 <= stream_id <= 0xDF:
                    audio.append(pts)
        pos += TS_PACKET_SIZE
    v, a = _earliest(video), _earliest(audio)
    if v is None and a is None:
        return None
    return TsInfo(
        video_pts=None if v is None else v / TS_CLOCK_HZ,
        audio_pts=None if a is None else a / TS_CLOCK_HZ,
    )


# ----------------------------------------------------------------------
# Fragmented MP4 (what a SABR source delivers)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Fmp4Info:
    """Decode time of the first fragment and the summed sample durations, in
    seconds, plus the timescale they were read with."""

    timescale: int
    start: float
    duration: float


def _boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yields (type, payload_start, box_end) for the boxes in data[start:end]."""
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def _child(data: bytes, start: int, end: int, path: tuple[bytes, ...]) -> tuple[int, int] | None:
    for kind, payload, box_end in _boxes(data, start, end):
        if kind == path[0]:
            return (payload, box_end) if len(path) == 1 else _child(data, payload, box_end, path[1:])
    return None


def fmp4_timescale(data: bytes) -> int | None:
    """The media timescale from a moov box (init segment or self-initializing fragment)."""
    for kind, payload, box_end in _boxes(data):
        if kind != b"moov":
            continue
        mdhd = _child(data, payload, box_end, (b"trak", b"mdia", b"mdhd"))
        if mdhd is None:
            return None
        p = mdhd[0]
        offset = p + (20 if data[p] == 1 else 12)
        if offset + 4 > mdhd[1]:
            return None
        timescale = struct.unpack_from(">I", data, offset)[0]
        return timescale or None
    return None


def _trex_default_duration(data: bytes) -> int:
    for kind, payload, box_end in _boxes(data):
        if kind == b"moov":
            trex = _child(data, payload, box_end, (b"mvex", b"trex"))
            if trex is not None and trex[0] + 16 <= trex[1]:
                # version/flags, track_ID, default_sample_description_index, default_sample_duration
                return struct.unpack_from(">I", data, trex[0] + 12)[0]
    return 0


def parse_fmp4(data: bytes, timescale: int | None = None) -> Fmp4Info | None:
    """Reads tfdt and trun of every moof. The timescale comes from the data's own
    moov when present (YouTube's live fragments carry one), else from the
    `timescale` argument (taken from a separate init segment)."""
    timescale = fmp4_timescale(data) or timescale
    if not timescale:
        return None
    trex_default = _trex_default_duration(data)
    start: int | None = None
    total = 0
    for kind, payload, box_end in _boxes(data):
        if kind != b"moof":
            continue
        traf = _child(data, payload, box_end, (b"traf",))
        if traf is None:
            continue
        default_duration = trex_default
        decode_time: int | None = None
        durations = 0
        for sub, p, sub_end in _boxes(data, traf[0], traf[1]):
            if sub == b"tfhd" and p + 8 <= sub_end:
                flags = int.from_bytes(data[p + 1 : p + 4], "big")
                off = p + 8  # version/flags + track_ID
                if flags & 0x01:
                    off += 8
                if flags & 0x02:
                    off += 4
                if flags & 0x08 and off + 4 <= sub_end:
                    default_duration = struct.unpack_from(">I", data, off)[0]
            elif sub == b"tfdt" and p + 8 <= sub_end:
                if data[p] == 1 and p + 12 <= sub_end:
                    decode_time = struct.unpack_from(">Q", data, p + 4)[0]
                else:
                    decode_time = struct.unpack_from(">I", data, p + 4)[0]
            elif sub == b"trun" and p + 8 <= sub_end:
                flags = int.from_bytes(data[p + 1 : p + 4], "big")
                count = struct.unpack_from(">I", data, p + 4)[0]
                off = p + 8
                if flags & 0x001:
                    off += 4
                if flags & 0x004:
                    off += 4
                per_sample = sum(4 for bit in (0x100, 0x200, 0x400, 0x800) if flags & bit)
                if flags & 0x100:
                    for i in range(count):
                        at = off + i * per_sample
                        if at + 4 > sub_end:
                            break
                        durations += struct.unpack_from(">I", data, at)[0]
                else:
                    durations += count * default_duration
        if decode_time is None:
            continue
        if start is None:
            start = decode_time
        total += durations
    if start is None:
        return None
    return Fmp4Info(timescale=timescale, start=start / timescale, duration=total / timescale)


# ----------------------------------------------------------------------
# Segment inspection
# ----------------------------------------------------------------------


def detect_container(data: bytes) -> str | None:
    if len(data) >= 8 and data[4:8] in _FMP4_BOX_TYPES:
        return CONTAINER_FMP4
    if data[:1] == bytes([TS_SYNC_BYTE]) and (len(data) <= TS_PACKET_SIZE or data[TS_PACKET_SIZE] == TS_SYNC_BYTE):
        return CONTAINER_MPEGTS
    pos = _id3_tag_length(data, 0)
    if len(data) - pos >= 2 and data[pos] == 0xFF and (data[pos + 1] & 0xF6) == 0xF0:
        return CONTAINER_ADTS
    return None


@dataclass
class TrackSegment:
    """One media segment of one track, as delivered by a SegmentSource.

    `clock` is the segment's own timestamp of its first sample in seconds, in the
    source's clock domain (for HLS the raw, wrapping 33-bit clock), or None when
    the segment carries none (the assembler may then derive it). `duration` is
    exact for ADTS and fmp4 and nominal for TS. `tolerance` is how far the next
    segment's clock may deviate from clock + duration and still count as
    continuous. `init` is the initialization segment a fragmented-MP4 track needs
    in front of its fragments when they aren't self-initializing.
    """

    track: str
    sequence: int
    data: bytes
    container: str
    clock: float | None
    duration: float
    tolerance: float = NOMINAL_TOLERANCE_SECONDS
    init: bytes | None = None


def inspect_segment(track: str, sequence: int, data: bytes, nominal_duration: float, init: bytes | None = None) -> TrackSegment | None:
    """Reads a downloaded segment's container, clock and duration from its own
    bytes. None if it can't be read. Every SegmentSource stamps through here."""
    container = detect_container(data)
    if container == CONTAINER_ADTS:
        adts = parse_adts(data)
        if adts is None:
            return None
        # A missing ID3 timestamp is allowed: the assembler derives it from the previous segment.
        tolerance = 512 / adts.sample_rate  # half an AAC frame
        return TrackSegment(track, sequence, data, CONTAINER_ADTS, parse_id3_timestamp(data), adts.duration, tolerance)
    if container == CONTAINER_MPEGTS:
        info = scan_ts_pts(data)
        if info is None:
            return None
        # Audio and muxed tracks are clocked by their audio: transcription offsets
        # count from the first audio sample, so a muxed segment without audio has no clock.
        clock = info.video_pts if track == TRACK_VIDEO else info.audio_pts
        if clock is None:
            return None
        return TrackSegment(track, sequence, data, CONTAINER_MPEGTS, clock, nominal_duration)
    if container == CONTAINER_FMP4:
        timescale = fmp4_timescale(init) if init else None
        fmp4 = parse_fmp4(data, timescale)
        if fmp4 is None or fmp4.duration <= 0:
            return None
        return TrackSegment(track, sequence, data, CONTAINER_FMP4, fmp4.start, fmp4.duration, 1.0 / fmp4.timescale + 0.002, init)
    return None


# ----------------------------------------------------------------------
# Clock
# ----------------------------------------------------------------------


class ClockUnwrapper:
    """Turns a wrapping clock into a monotonic one. `wrap_period` None means the
    clock doesn't wrap (fmp4). The wrap count is seeded on the first value from
    `expected`, the clock the stream should be near (time since its start), so a
    run that begins after a wrap (a long stream, a restart) still lands on the
    right side of it."""

    def __init__(self, wrap_period: float | None = TS_WRAP_SECONDS, expected: float | None = None):
        self.wrap_period = wrap_period
        self._expected = expected
        self._last_raw: float | None = None
        self._offset = 0.0
        self.wrapped = False  # set when the most recent unwrap() crossed a wrap

    @property
    def offset(self) -> float:
        return self._offset

    def unwrap(self, raw: float) -> float:
        self.wrapped = False
        period = self.wrap_period
        if period is None:
            return raw
        if self._last_raw is None:
            if self._expected is not None:
                self._offset = max(0, round((self._expected - raw) / period)) * period
        elif raw < self._last_raw - period / 2:
            self._offset += period
            self.wrapped = True
        elif raw > self._last_raw + period / 2:
            # A late value from before the wrap: map it back without moving the reference.
            return raw + self._offset - period
        self._last_raw = raw
        return raw + self._offset

    def to_raw(self, clock: float) -> float:
        if self.wrap_period is None:
            return clock
        return clock % self.wrap_period


# ----------------------------------------------------------------------
# Chunks
# ----------------------------------------------------------------------


@dataclass
class SegmentGroup:
    """All tracks of one sequence number. A track that could not be fetched or
    read is absent from `segments`."""

    sequence: int
    segments: dict[str, TrackSegment] = field(default_factory=dict)


@dataclass
class ChunkTrack:
    container: str
    clock: float  # this track's first clock in the chunk, in the source's (raw) domain: what the mux uses
    init: bytes | None
    data: bytearray = field(default_factory=bytearray)


@dataclass
class TimelineChunk:
    """Consecutive segments ready to be muxed and transcribed. `clock` is the
    unwrapped clock of the first sample of the clock track: the chunk's VOD
    position (before the source's vod_offset). `duration` is the clock track's.

    Not to be confused with models.Chunk: that is what one of these becomes
    once muxed to a file and decoded, the unit the pipeline transcribes."""

    first_sequence: int
    last_sequence: int
    clock: float
    duration: float
    tracks: dict[str, ChunkTrack] = field(default_factory=dict)


class ChunkAssembler:
    """Groups consecutive SegmentGroups into chunks of about `target_seconds`.

    The clock track (audio, or the muxed track when there is no separate audio)
    defines the chunk's time: transcription runs on it, the transcript's offsets
    are relative to its first sample, and raw ADTS is laid out back to back from
    its first timestamp. So a chunk holds exactly one unbroken run of it, and ends
    early on any break: a group without the clock track, a skipped sequence, a
    clock that doesn't continue where the previous segment ended (within the
    segment's tolerance), a clock wrap, or a changed format. Other tracks may
    have holes (a missing video segment leaves a gap in the video only, which
    -copyts muxing keeps in place).

    A clock-track segment without a timestamp continues the run at the expected
    clock (exact for contiguous audio), or starts a run at its video segment's
    timestamp. Groups at or before the last accepted sequence are duplicates
    and are dropped.
    """

    # Clock-track segments in a row that may borrow their clock before the run is broken.
    MAX_DERIVED_CLOCKS = 10

    def __init__(
        self,
        target_seconds: float,
        clock_track: str,
        unwrapper: ClockUnwrapper | None = None,
        on_gap: Callable[[str], None] | None = None,
    ):
        self.target_seconds = target_seconds
        self.clock_track = clock_track
        self.unwrapper = unwrapper or ClockUnwrapper()
        self.on_gap = on_gap  # called with a human-readable reason when a run of the clock track breaks
        self.duplicates = 0
        self.derived_clocks = 0
        self._current: TimelineChunk | None = None
        self._last_sequence: int | None = None  # last sequence of the current run
        self._last_accepted: int | None = None  # last sequence ever accepted (duplicate guard)
        self._expected_clock: float | None = None  # unwrapped clock where the run continues
        self._derived_in_a_row = 0

    @property
    def pending_duration(self) -> float:
        return self._current.duration if self._current else 0.0

    def add(self, group: SegmentGroup) -> list[TimelineChunk]:
        out: list[TimelineChunk] = []
        if self._last_accepted is not None and group.sequence <= self._last_accepted:
            self.duplicates += 1
            return out

        clock_seg = group.segments.get(self.clock_track)
        clock = self._clock_of(group, clock_seg)
        if clock_seg is None or clock is None:
            reason = f"sequence {group.sequence} has no {'usable ' if clock_seg else ''}{self.clock_track} segment"
            self._break(out, reason)
            return out

        reason = None
        if self._current is not None:
            if self.unwrapper.wrapped:
                reason = "clock wrapped"
            elif self._last_sequence is not None and group.sequence != self._last_sequence + 1:
                reason = f"sequence jumped {self._last_sequence} -> {group.sequence}"
            elif self._expected_clock is not None and abs(clock - self._expected_clock) > clock_seg.tolerance:
                reason = f"clock jumped {clock - self._expected_clock:+.3f}s at sequence {group.sequence}"
            else:
                for name, seg in group.segments.items():
                    track = self._current.tracks.get(name)
                    if track is not None and (track.container != seg.container or track.init != seg.init):
                        reason = f"{name} format changed at sequence {group.sequence}"
                        break
        if reason is not None:
            self._gap(reason)
            out.append(self._close())

        if self._current is None:
            self._current = TimelineChunk(first_sequence=group.sequence, last_sequence=group.sequence, clock=clock, duration=0.0)
        chunk = self._current
        for name, seg in group.segments.items():
            track = chunk.tracks.get(name)
            if track is None:
                if name == self.clock_track:
                    raw = self.unwrapper.to_raw(clock)
                else:
                    raw = seg.clock if seg.clock is not None else self.unwrapper.to_raw(clock)
                track = chunk.tracks[name] = ChunkTrack(container=seg.container, clock=raw, init=seg.init)
            track.data += seg.data
        chunk.last_sequence = group.sequence
        chunk.duration += clock_seg.duration
        self._last_sequence = group.sequence
        self._last_accepted = group.sequence
        self._expected_clock = clock + clock_seg.duration

        # Close at the segment boundary nearest the target: with 5 s segments a 6 s
        # target gives 5 s chunks rather than 10 s ones.
        if chunk.duration + clock_seg.duration / 2 >= self.target_seconds:
            out.append(self._close())
        return out

    def flush(self) -> TimelineChunk | None:
        """Returns the partial chunk, if any. The next group starts a new run."""
        chunk = self._close() if self._current is not None else None
        self._last_sequence = None
        self._expected_clock = None
        return chunk

    def _clock_of(self, group: SegmentGroup, clock_seg: TrackSegment | None) -> float | None:
        """Unwrapped clock of the group's clock-track segment, derived when it carries none."""
        if clock_seg is None:
            return None
        if clock_seg.clock is not None:
            self._derived_in_a_row = 0
            return self.unwrapper.unwrap(clock_seg.clock)
        self.unwrapper.wrapped = False
        # _expected_clock and _last_sequence survive a chunk closing at its target (the
        # run continues) and are cleared on any break, so this holds across chunks too.
        if (
            self._expected_clock is not None
            and self._last_sequence is not None
            and group.sequence == self._last_sequence + 1
            and self._derived_in_a_row < self.MAX_DERIVED_CLOCKS
        ):
            self._derived_in_a_row += 1
            self.derived_clocks += 1
            return self._expected_clock
        for name, seg in group.segments.items():
            if name != self.clock_track and seg.clock is not None:
                # Video segment N starts within milliseconds of audio segment N.
                self._derived_in_a_row += 1
                self.derived_clocks += 1
                return self.unwrapper.unwrap(seg.clock)
        return None

    def _break(self, out: list[TimelineChunk], reason: str) -> None:
        if self._current is not None:
            self._gap(reason)
            out.append(self._close())
        # Whatever follows can't continue the run that just broke.
        self._last_sequence = None
        self._expected_clock = None

    def _close(self) -> TimelineChunk:
        chunk = self._current
        assert chunk is not None
        self._current = None
        return chunk

    def _gap(self, reason: str) -> None:
        if self.on_gap is not None:
            self.on_gap(reason)


# ----------------------------------------------------------------------
# Muxing
# ----------------------------------------------------------------------


class MuxError(Exception):
    pass


_INPUT_FORMAT = {CONTAINER_MPEGTS: "mpegts", CONTAINER_ADTS: "aac", CONTAINER_FMP4: "mp4"}
_INPUT_SUFFIX = {CONTAINER_MPEGTS: ".ts", CONTAINER_ADTS: ".aac", CONTAINER_FMP4: ".mp4"}


def build_mux_command(ffmpeg_path: str, inputs: list[tuple[str, ChunkTrack, str]], output: str) -> list[str]:
    """ffmpeg command muxing the given (track name, track, file path) inputs into
    an MPEG-TS file at `output`, keeping the source clock on every stream.

    -copyts keeps each input's own timestamps instead of rebasing every input
    to zero (which is what puts split HLS video and audio out of sync). Raw ADTS
    carries no timestamps, so an ADTS input is placed on the clock with
    -itsoffset (its first frame's clock; the rest follow by sample count).
    -correct_ts_overflow 0 stops the TS demuxer from shifting timestamps that
    start just below the 33-bit wrap to negative values.
    """
    cmd = [ffmpeg_path, "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-copyts"]
    maps: list[str] = []
    for index, (name, track, path) in enumerate(inputs):
        if track.container == CONTAINER_ADTS:
            cmd += ["-itsoffset", f"{track.clock:.6f}"]
        elif track.container == CONTAINER_MPEGTS:
            cmd += ["-correct_ts_overflow", "0"]
        cmd += ["-f", _INPUT_FORMAT[track.container], "-i", path]
        if name in (TRACK_VIDEO, TRACK_MUXED):
            maps += ["-map", f"{index}:v:0?"]
        if name in (TRACK_AUDIO, TRACK_MUXED):
            maps += ["-map", f"{index}:a:0?"]
    return [*cmd, *maps, "-c", "copy", "-f", "mpegts", output]


async def mux_chunk(ffmpeg_path: str, chunk: TimelineChunk, workdir: Path, output: Path, timeout: float = 30.0) -> None:
    """Muxes a chunk's tracks into one MPEG-TS file at `output`. Raises MuxError.

    Each track's (init +) data goes to its own temp file in `workdir` first
    (ffmpeg takes one input per track, and only one could come through stdin).
    Those temp files are always removed. On any failure (including cancellation)
    `output` is removed too, so a file at `output` is always a complete mux."""
    paths: list[str] = []
    ok = False
    try:
        if not any(track.data for track in chunk.tracks.values()):
            raise MuxError("chunk has no media")
        inputs: list[tuple[str, ChunkTrack, str]] = []
        try:
            workdir.mkdir(parents=True, exist_ok=True)
            for name in (TRACK_VIDEO, TRACK_MUXED, TRACK_AUDIO):
                track = chunk.tracks.get(name)
                if track is None or not track.data:
                    continue
                fd, path = tempfile.mkstemp(dir=workdir, prefix=f"{chunk.first_sequence}-{name}-", suffix=_INPUT_SUFFIX[track.container])
                paths.append(path)
                with os.fdopen(fd, "wb") as f:
                    if track.init:
                        f.write(track.init)
                    f.write(track.data)
                inputs.append((name, track, path))
        except OSError as e:
            raise MuxError(f"cannot write the mux inputs: {e}") from e
        cmd = build_mux_command(ffmpeg_path, inputs, str(output))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            raise MuxError(f"ffmpeg could not be started: {e}") from e
        try:
            _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            raise MuxError(f"ffmpeg timed out after {timeout:.0f}s") from None
        finally:
            # Timed out or cancelled: ffmpeg must not outlive the call (it would
            # keep writing `output` and holding the inputs we are about to delete).
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        try:
            size = output.stat().st_size
        except OSError:
            size = 0
        if proc.returncode != 0 or not size:
            stderr = (stderr_bytes or b"").decode(errors="replace").strip()
            raise MuxError(f"ffmpeg exit={proc.returncode}: {stderr[-2000:] or 'no output'}")
        ok = True
    finally:
        for path in paths:
            with contextlib.suppress(OSError):
                os.remove(path)
        if not ok:
            with contextlib.suppress(OSError):
                output.unlink(missing_ok=True)
