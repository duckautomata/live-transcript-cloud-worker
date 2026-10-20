"""Byte-level builders for YouTube-style live segments, shared by the
YouTube Timeline tests: ID3-timestamped ADTS audio and MPEG-TS packets
carrying a PES PTS."""

from __future__ import annotations

from live_transcript_cloud_worker.youtube_timeline import ID3_TIMESTAMP_OWNER, TS_CLOCK_HZ, TS_WRAP_TICKS

ADTS_44100_INDEX = 4
SAMPLES_PER_FRAME = 1024


def _syncsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def id3_timestamp_tag(seconds: float) -> bytes:
    """An ID3v2.4 tag with the PRIV frame HLS uses to timestamp raw audio segments."""
    ticks = round(seconds * TS_CLOCK_HZ) % TS_WRAP_TICKS
    priv = ID3_TIMESTAMP_OWNER + ticks.to_bytes(8, "big")
    frame = b"PRIV" + _syncsafe(len(priv)) + b"\x00\x00" + priv
    return b"ID3\x04\x00\x00" + _syncsafe(len(frame)) + frame


def adts_frame(payload: bytes = b"\x21" * 16, sr_index: int = ADTS_44100_INDEX, blocks: int = 0) -> bytes:
    """One ADTS frame (AAC LC, stereo, no CRC) around an arbitrary payload."""
    length = 7 + len(payload)
    header = bytes(
        [
            0xFF,
            0xF1,
            (1 << 6) | (sr_index << 2),
            (2 << 6) | ((length >> 11) & 0x03),
            (length >> 3) & 0xFF,
            ((length & 0x07) << 5) | 0x1F,
            0xFC | blocks,
        ]
    )
    return header + payload


def audio_segment(clock: float, frames: int = 43, sr_index: int = ADTS_44100_INDEX) -> bytes:
    """A raw AAC HLS segment: ID3 timestamp tag + `frames` ADTS frames."""
    return id3_timestamp_tag(clock) + b"".join(adts_frame(bytes([i % 256]) * 16, sr_index) for i in range(frames))


def audio_duration(frames: int = 43, rate: int = 44100) -> float:
    return frames * SAMPLES_PER_FRAME / rate


def _pts_bytes(ticks: int) -> bytes:
    return bytes(
        [
            0x21 | ((ticks >> 29) & 0x0E),
            (ticks >> 22) & 0xFF,
            ((ticks >> 14) & 0xFE) | 0x01,
            (ticks >> 7) & 0xFF,
            ((ticks << 1) & 0xFE) | 0x01,
        ]
    )


def ts_pes_packet(pid: int, stream_id: int, seconds: float) -> bytes:
    """A 188-byte TS packet starting a PES with the given PTS."""
    ticks = round(seconds * TS_CLOCK_HZ) % TS_WRAP_TICKS
    pes = b"\x00\x00\x01" + bytes([stream_id]) + b"\x00\x00" + b"\x80\x80\x05" + _pts_bytes(ticks)
    header = bytes([0x47, 0x40 | ((pid >> 8) & 0x1F), pid & 0xFF, 0x10])
    return header + pes + b"\xff" * (184 - len(pes))


def ts_filler_packet(pid: int) -> bytes:
    return bytes([0x47, (pid >> 8) & 0x1F, pid & 0xFF, 0x10]) + b"\xff" * 184


def video_segment(clock: float, frames: int = 3, frame_interval: float = 1 / 30) -> bytes:
    """A video TS segment whose PES PTS values start at `clock`."""
    packets = []
    for i in range(frames):
        packets.append(ts_pes_packet(0x100, 0xE0, clock + i * frame_interval))
        packets.append(ts_filler_packet(0x100))
    return b"".join(packets)


# ----------------------------------------------------------------------
# Fragmented MP4
# ----------------------------------------------------------------------


def _box(kind: bytes, payload: bytes) -> bytes:
    import struct

    return struct.pack(">I", 8 + len(payload)) + kind + payload


def fmp4_init(timescale: int = 44100, trex_default: int = 1024) -> bytes:
    """ftyp + a minimal moov carrying mdhd's timescale and trex's default sample duration."""
    import struct

    mdhd = _box(b"mdhd", b"\x00\x00\x00\x00" + struct.pack(">III", 0, 0, timescale) + struct.pack(">I", 0) + b"\x55\xc4\x00\x00")
    trak = _box(b"trak", _box(b"mdia", mdhd))
    trex = _box(b"trex", b"\x00\x00\x00\x00" + struct.pack(">IIIII", 1, 1, trex_default, 0, 0))
    moov = _box(b"moov", _box(b"mvex", trex) + trak)
    return _box(b"ftyp", b"iso6\x00\x00\x00\x00") + moov


def fmp4_fragment(
    decode_time: int,
    durations: list[int] | None = None,
    count: int | None = None,
    default_duration: int | None = None,
    version: int = 1,
) -> bytes:
    """moof (tfhd, tfdt, trun) + mdat. Per-sample durations when given, else
    `count` samples using tfhd's (or trex's) default duration."""
    import struct

    tfhd_flags = 0x08 if default_duration is not None else 0
    tfhd = _box(
        b"tfhd", struct.pack(">I", tfhd_flags) + struct.pack(">I", 1) + (struct.pack(">I", default_duration) if default_duration else b"")
    )
    if version == 1:
        tfdt = _box(b"tfdt", b"\x01\x00\x00\x00" + struct.pack(">Q", decode_time))
    else:
        tfdt = _box(b"tfdt", b"\x00\x00\x00\x00" + struct.pack(">I", decode_time))
    if durations is not None:
        trun = _box(
            b"trun", struct.pack(">I", 0x100) + struct.pack(">I", len(durations)) + b"".join(struct.pack(">I", d) for d in durations)
        )
    else:
        trun = _box(b"trun", struct.pack(">I", 0) + struct.pack(">I", count or 0))
    moof = _box(b"moof", _box(b"mfhd", b"\x00\x00\x00\x00\x00\x00\x00\x01") + _box(b"traf", tfhd + tfdt + trun))
    return moof + _box(b"mdat", b"\x00" * 16)


def shift_tfdt(data: bytes, seconds: float) -> bytes:
    """Moves every fragment of a fragmented MP4 by `seconds` (ffmpeg always writes them from 0)."""
    import struct

    from live_transcript_cloud_worker.youtube_timeline import _boxes, _child, fmp4_timescale

    out = bytearray(data)
    timescale = fmp4_timescale(data)
    assert timescale
    delta = round(seconds * timescale)
    for kind, payload, end in _boxes(data):
        if kind != b"moof":
            continue
        traf = _child(data, payload, end, (b"traf",))
        assert traf is not None
        for sub, p, _ in _boxes(data, traf[0], traf[1]):
            if sub == b"tfdt":
                if out[p] == 1:
                    struct.pack_into(">Q", out, p + 4, struct.unpack_from(">Q", out, p + 4)[0] + delta)
                else:
                    struct.pack_into(">I", out, p + 4, struct.unpack_from(">I", out, p + 4)[0] + delta)
    return bytes(out)
