"""
ponemah_reader.py
=================
Read DSI Ponemah 6.x experiments in PURE PYTHON -- no DSI DLLs, no .NET.

A Ponemah experiment folder contains:
  *.PnmWav    binary waveform data (TLV: tag/length/value records)
  *.PnmExp    SQLite config database (channel labels, units, calibration)
  *.PnmMarks  SQLite database whose ChannelContextInfo table gives each
              channel's TicksPerSample (=> sample rate)

The .PnmWav TLV layout and the ECG calibration used here were validated by
reproducing DSI Ponemah's own exported values exactly (bit-for-bit on the ECG
channel against a Ponemah reference CSV).

Channel mapping (validated on real data):
  PnmWav channel_id k  <->  ChannelContextInfo.SubjectChannelId k
                       <->  .PnmExp CHANNEL config number (k - 1)

Calibration:
  ECG channels (config unit 'mV'): millivolts = raw_int16 * 10/32256
  Other channels: exported as raw ADC counts (their physical scale factors are
  not stored in a documented form; ECG is the validated, calibrated signal).
"""

from __future__ import annotations

import glob
import os
import sqlite3
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterator, List, Optional

import numpy as np

# .NET ticks: 100-ns units since 0001-01-01.
TICKS_PER_SECOND = 10_000_000
DOTNET_EPOCH = datetime(1, 1, 1)
ECG_MV_PER_COUNT = 10.0 / 32256.0  # validated against Ponemah


# --------------------------------------------------------------------------- #
# Data model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    start_utc: datetime
    end_utc: datetime

    @property
    def duration_s(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds()


@dataclass
class ChannelInfo:
    channel_id: int
    label: str
    sample_rate: float
    unit: str = "mV"
    scale: float = 1.0          # multiply raw int16 by this to get physical units
    n_samples: int = 0
    digital: bool = False     # samples are raw int16 ADC counts
    segments: List[Segment] = field(default_factory=list)


@dataclass
class SubjectInfo:
    subject_id: int
    name: str
    channels: List[ChannelInfo] = field(default_factory=list)


class PonemahReader:
    def list_subjects(self) -> List[SubjectInfo]:
        raise NotImplementedError

    def read_channel_segment(self, channel_id, segment, max_chunk=2_000_000):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# .PnmWav TLV parsing                                                         #
# --------------------------------------------------------------------------- #
class _PnmWav:
    MAGIC = b"\xAF\x55\x55\xAA"
    TAG_CHANNEL_ID = 0x13B5
    TAG_DATA_BLOCK = 0x1397
    TAG_START_TIME = 0x1395
    TAG_END_TIME = 0x1396
    TAG_SUBJECT = 0x13B1
    TAG_TIMEZONE = 0x13B2

    def __init__(self, path: str):
        self.path = path
        self.start_ticks: Optional[int] = None
        self.subject_name: Optional[str] = None
        self.timezone: Optional[str] = None
        with open(path, "rb") as f:
            if f.read(4) != self.MAGIC:
                raise ValueError(f"{os.path.basename(path)} is not a PnmWav file")

    def scan(self):
        """One fast pass: per-channel sample counts + header fields."""
        counts: dict[int, int] = {}
        last = None
        with open(self.path, "rb") as f:
            f.read(4)
            while True:
                tb = f.read(2)
                if len(tb) < 2:
                    break
                tag = struct.unpack("<H", tb)[0]
                size = struct.unpack("<I", f.read(4))[0]
                if tag == self.TAG_CHANNEL_ID:
                    last = struct.unpack("<Q", f.read(size))[0]
                elif tag == self.TAG_DATA_BLOCK:
                    counts[last] = counts.get(last, 0) + size // 2
                    f.seek(size, 1)
                elif tag == self.TAG_START_TIME:
                    v = f.read(size)
                    if self.start_ticks is None:
                        self.start_ticks = struct.unpack("<q", v)[0]
                elif tag == self.TAG_SUBJECT:
                    self.subject_name = f.read(size).decode("utf-8", "ignore").strip("\x00")
                elif tag == self.TAG_TIMEZONE:
                    self.timezone = f.read(size).decode("utf-8", "ignore").strip("\x00")
                else:
                    f.seek(size, 1) if size >= 10_000_000 else f.read(size)
        return counts

    def iter_channel(self, channel_id: int) -> Iterator[np.ndarray]:
        """Yield int16 sample chunks for one channel, skipping all others."""
        last = None
        with open(self.path, "rb") as f:
            f.read(4)
            while True:
                tb = f.read(2)
                if len(tb) < 2:
                    break
                tag = struct.unpack("<H", tb)[0]
                size = struct.unpack("<I", f.read(4))[0]
                if tag == self.TAG_CHANNEL_ID:
                    last = struct.unpack("<Q", f.read(size))[0]
                elif tag == self.TAG_DATA_BLOCK:
                    if last == channel_id:
                        yield np.frombuffer(f.read(size), dtype="<i2")
                    else:
                        f.seek(size, 1)
                else:
                    f.seek(size, 1) if size >= 10_000_000 else f.read(size)

    @property
    def start_datetime(self) -> Optional[datetime]:
        if self.start_ticks is None:
            return None
        return DOTNET_EPOCH + timedelta(microseconds=self.start_ticks / 10)


# --------------------------------------------------------------------------- #
# Metadata from the SQLite sidecars                                           #
# --------------------------------------------------------------------------- #
def find_experiment_files(exp_folder: str):
    def g(ext):
        return (glob.glob(os.path.join(exp_folder, f"*{ext}")) +
                glob.glob(os.path.join(exp_folder, f"*{ext.lower()}")))
    wav = g(".PnmWav")
    exp = g(".PnmExp")
    marks = g(".PnmMarks")
    return (wav[0] if wav else None,
            exp[0] if exp else None,
            marks[0] if marks else None)


def _rates_from_marks(marks_path: str) -> dict[int, float]:
    out = {}
    if not marks_path or not os.path.exists(marks_path):
        return out
    con = sqlite3.connect(f"file:{marks_path}?mode=ro", uri=True)
    try:
        for ctx, subj_ch, tps in con.execute(
            "SELECT ChannelContextId, SubjectChannelId, TicksPerSample "
            "FROM ChannelContextInfo"
        ):
            if tps:
                out[int(subj_ch)] = TICKS_PER_SECOND / float(tps)
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return out


def _labels_from_exp(exp_path: str) -> dict[int, tuple]:
    """Return {config_channel_number: (label, unit)} from the CHANNEL section."""
    out = {}
    if not exp_path or not os.path.exists(exp_path):
        return out
    con = sqlite3.connect(f"file:{exp_path}?mode=ro", uri=True)
    try:
        for meta, payload in con.execute(
            'SELECT Metadata, Payload FROM ConfigInfo WHERE ConfigSection="CHANNEL"'
        ):
            try:
                num = int(meta)
            except (TypeError, ValueError):
                continue
            p = payload if isinstance(payload, str) else \
                (payload.decode("utf-8", "ignore") if payload else "")
            lines = p.split("\r\n")
            if len(lines) > 4:
                out[num] = (lines[3].strip(), lines[4].strip())
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return out


# --------------------------------------------------------------------------- #
# The real reader                                                             #
# --------------------------------------------------------------------------- #
class PurePythonPonemahReader(PonemahReader):
    def __init__(self, exp_folder: str, include_aux: bool = True, verbose=False,
                 max_seconds: float | None = None):
        self.exp_folder = exp_folder
        self.include_aux = include_aux
        self.verbose = verbose
        self.max_seconds = max_seconds
        wav, exp, marks = find_experiment_files(exp_folder)
        if not wav:
            raise FileNotFoundError(f"No .PnmWav file found in {exp_folder}")
        self.wav = _PnmWav(wav)
        self._exp_path, self._marks_path = exp, marks
        self._subjects: Optional[List[SubjectInfo]] = None

    def list_subjects(self) -> List[SubjectInfo]:
        if self._subjects is not None:
            return self._subjects
        counts = self.wav.scan()
        rates = _rates_from_marks(self._marks_path)
        labels = _labels_from_exp(self._exp_path)
        start = self.wav.start_datetime or datetime(2000, 1, 1)

        channels = []
        for cid in sorted(counts):
            rate = rates.get(cid, 0.0)
            label, unit = labels.get(cid - 1, (f"CH{cid}", ""))  # config number = id-1
            label = label or f"CH{cid}"
            if unit.lower() == "mv":
                scale, out_unit = ECG_MV_PER_COUNT, "mV"
            else:
                scale, out_unit = 1.0, "count"   # uncalibrated aux channel
            is_ecg = unit.lower() == "mv"
            if not is_ecg and not self.include_aux:
                continue
            n = counts[cid]
            if self.max_seconds and rate:
                n = min(n, int(self.max_seconds * rate))
            seg = Segment(start, start + timedelta(seconds=(n / rate if rate else 0)))
            ch = ChannelInfo(cid, label, rate, out_unit, scale, n, True, [seg])
            channels.append(ch)

        name = self.wav.subject_name or "Subject"
        self._subjects = [SubjectInfo(0, name, channels)]
        if self.verbose:
            print(f"Subject {name}: " + ", ".join(
                f"{c.label}({c.sample_rate:g}Hz,{c.unit})" for c in channels))
        return self._subjects

    def read_channel_segment(self, channel_id, segment, max_chunk=2_000_000):
        cap = None
        if self.max_seconds:
            for s2 in self.list_subjects():
                for c in s2.channels:
                    if c.channel_id == channel_id and c.sample_rate:
                        cap = int(self.max_seconds * c.sample_rate)
        buf, total, emitted = [], 0, 0
        for chunk in self.wav.iter_channel(channel_id):
            if cap is not None and emitted + total + len(chunk) > cap:
                chunk = chunk[: max(0, cap - emitted - total)]
            buf.append(chunk); total += len(chunk)
            if total >= max_chunk:
                yield np.concatenate(buf)        # raw int16 counts
                emitted += total; buf, total = [], 0
            if cap is not None and emitted + total >= cap:
                break
        if buf:
            yield np.concatenate(buf)


# --------------------------------------------------------------------------- #
# Mock reader (synthetic; used by tests / demo)                               #
# --------------------------------------------------------------------------- #
class MockPonemahReader(PonemahReader):
    def __init__(self, n_subjects=1, duration_s=30, with_gap=True):
        self._subjects = []
        base = datetime(2026, 6, 23, 9, 0, 0)
        for s in range(n_subjects):
            chans = [
                ChannelInfo(s * 10 + 1, "ECG", 1000.0, "mV"),
                ChannelInfo(s * 10 + 2, "BP", 500.0, "mmHg"),
                ChannelInfo(s * 10 + 3, "Temp", 1.0, "degC"),
            ]
            for ch in chans:
                ch.segments = [Segment(base, base + timedelta(seconds=duration_s))]
                if with_gap:
                    g0 = base + timedelta(seconds=duration_s + 300)
                    ch.segments.append(Segment(g0, g0 + timedelta(seconds=duration_s)))
            self._subjects.append(SubjectInfo(s, f"Subject_{s}", chans))

    def list_subjects(self):
        return self._subjects

    def read_channel_segment(self, channel_id, segment, max_chunk=2_000_000):
        rate = label = None
        for sub in self._subjects:
            for ch in sub.channels:
                if ch.channel_id == channel_id:
                    rate, label = ch.sample_rate, ch.label
        if rate is None:
            raise KeyError(channel_id)
        n = int(round(rate * segment.duration_s))
        t = np.arange(n) / rate
        if label == "ECG":
            sig = 1.2 * np.sin(2 * np.pi * 6 * t) + 0.15 * np.sin(2 * np.pi * 50 * t)
        elif label == "BP":
            sig = 100 + 20 * np.sin(2 * np.pi * 6 * t)
        else:
            sig = 37 + 0.2 * np.sin(2 * np.pi * 0.01 * t)
        for i in range(0, n, max_chunk):
            yield sig[i:i + max_chunk]
