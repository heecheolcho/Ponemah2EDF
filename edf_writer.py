"""
edf_writer.py  --  PURE-PYTHON EDF / EDF+ writer (no third-party deps but numpy)
================================================================================
Writes standards-compliant EDF (and EDF+C with annotations) files.

EDF stores each signal as little-endian 16-bit integers ("digital" values) plus
per-signal physical min/max so readers can recover physical units. For Ponemah
we pass the ORIGINAL ADC counts as digital values with the digital range set to
the source range (-32256..32255), so ECG is stored losslessly and the physical
range (+/-10 mV) reproduces Ponemah's calibration exactly.

Reference: Kemp & Olivan, EDF/EDF+ specification (edfplus.info).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from typing import Sequence

import numpy as np


@dataclass
class Channel:
    label: str
    samples: np.ndarray
    sample_rate: float
    physical_dimension: str = "mV"
    physical_min: float | None = None
    physical_max: float | None = None
    prefilter: str = ""
    transducer: str = ""
    digital: bool = False          # samples are raw integer ADC counts
    digital_min: int = -32768
    digital_max: int = 32767

    def __post_init__(self):
        dtype = np.int64 if self.digital else np.float64
        self.samples = np.asarray(self.samples, dtype=dtype).ravel()


def _record_duration_seconds(rates: Sequence[float]) -> float:
    if all(float(r).is_integer() for r in rates):
        return 1.0
    fracs = [Fraction(r).limit_denominator(1_000_000) for r in rates]

    def lcm(a, b):
        return a * b // math.gcd(a, b)

    denom_lcm, num_gcd = 1, 0
    for f in fracs:
        denom_lcm = lcm(denom_lcm, f.denominator)
        num_gcd = math.gcd(num_gcd, f.numerator)
    return 1.0 if num_gcd == 0 else float(Fraction(denom_lcm, num_gcd))


def _fld(value, width: int) -> bytes:
    """ASCII field, left-justified, space-padded/truncated to `width` bytes."""
    s = str(value)
    b = s.encode("ascii", "replace")[:width]
    return b + b" " * (width - len(b))


def _num8(x: float) -> str:
    """Format a number to fit EDF's 8-char numeric field with max precision."""
    if float(x).is_integer() and -9_999_999 <= x <= 99_999_999:
        return str(int(x))
    for dec in range(6, -1, -1):
        s = f"{x:.{dec}f}"
        if len(s) <= 8:
            return s
    return f"{x:.0f}"[:8]


def _phys_range(c: Channel):
    data = c.samples
    if c.digital:
        pmin, pmax = float(c.physical_min), float(c.physical_max)
        dmin, dmax = int(c.digital_min), int(c.digital_max)
    else:
        if data.size:
            pmin = float(np.min(data)) if c.physical_min is None else c.physical_min
            pmax = float(np.max(data)) if c.physical_max is None else c.physical_max
        else:
            pmin, pmax = -1.0, 1.0
        if pmin == pmax:
            pmin -= 1.0
            pmax += 1.0
        dmin, dmax = -32767, 32767
    return pmin, pmax, dmin, dmax


def _to_digital(c: Channel, pmin, pmax, dmin, dmax) -> np.ndarray:
    if c.digital:
        d = c.samples
    else:
        scale = (dmax - dmin) / (pmax - pmin)
        d = np.round((c.samples - pmin) * scale + dmin)
    return np.clip(d, dmin, dmax).astype("<i2")


def write_edf(out_path, channels, start_time=None, patient_code="",
              patient_name="", recording_additional="", equipment="DSI Ponemah",
              annotations=None):
    if not channels:
        raise ValueError("No channels to write.")
    channels = list(channels)
    rates = [c.sample_rate for c in channels]
    rec_dur = _record_duration_seconds(rates)

    spr = [int(round(c.sample_rate * rec_dur)) for c in channels]
    n_records = min(int(c.samples.size // s) for c, s in zip(channels, spr) if s)
    if n_records == 0:
        raise ValueError(f"Not enough samples for one data record (dur={rec_dur}s).")

    # Precompute physical ranges + digital arrays
    ranges = [_phys_range(c) for c in channels]
    digitals = [_to_digital(c, *r)[: n_records * s]
                for c, r, s in zip(channels, ranges, spr)]

    use_plus = bool(annotations)
    ann_spr = 0
    ann_records = None
    if use_plus:
        # Build one TAL bytestring per data record (timekeeping + any events).
        ann_records = [bytearray() for _ in range(n_records)]
        for r in range(n_records):
            onset = r * rec_dur
            ann_records[r] += _tal(onset, None, None)  # timekeeping TAL
        for onset, dur, text in annotations:
            r = min(int(onset // rec_dur), n_records - 1)
            ann_records[r] += _tal(onset, dur, text)
        maxlen = max(len(a) for a in ann_records)
        ann_spr = max(8, (maxlen + 1) // 2 + 1)  # int16 units (2 bytes each)

    ns = len(channels) + (1 if use_plus else 0)
    header_bytes = 256 * (ns + 1)

    with open(out_path, "wb") as f:
        # ---- main header ----
        f.write(_fld("0", 8))
        pid = " ".join([_clean(patient_code) or "X", "X", "X",
                        _clean(patient_name) or "X"])
        f.write(_fld(pid, 80))
        sd = (start_time or datetime(1985, 1, 1))
        rid = "Startdate " + sd.strftime("%d-%b-%Y").upper() + " X X " + \
              (_clean(equipment) or "X")
        if recording_additional:
            rid += " " + _clean(recording_additional)
        f.write(_fld(rid, 80))
        f.write(_fld(sd.strftime("%d.%m.%y"), 8))
        f.write(_fld(sd.strftime("%H.%M.%S"), 8))
        f.write(_fld(header_bytes, 8))
        f.write(_fld("EDF+C" if use_plus else "", 44))
        f.write(_fld(n_records, 8))
        f.write(_fld(_num8(rec_dur), 8))
        f.write(_fld(ns, 4))

        # ---- signal headers ----
        labels = [c.label for c in channels] + (["EDF Annotations"] if use_plus else [])
        for lab in labels:
            f.write(_fld(lab, 16))
        for c in channels:
            f.write(_fld(c.transducer, 80))
        if use_plus:
            f.write(_fld("", 80))
        for c in channels:
            f.write(_fld(c.physical_dimension, 8))
        if use_plus:
            f.write(_fld("", 8))
        for (pmin, pmax, dmin, dmax) in ranges:
            f.write(_fld(_num8(pmin), 8))
        if use_plus:
            f.write(_fld("-1", 8))
        for (pmin, pmax, dmin, dmax) in ranges:
            f.write(_fld(_num8(pmax), 8))
        if use_plus:
            f.write(_fld("1", 8))
        for (pmin, pmax, dmin, dmax) in ranges:
            f.write(_fld(dmin, 8))
        if use_plus:
            f.write(_fld("-32768", 8))
        for (pmin, pmax, dmin, dmax) in ranges:
            f.write(_fld(dmax, 8))
        if use_plus:
            f.write(_fld("32767", 8))
        for c in channels:
            f.write(_fld(c.prefilter, 80))
        if use_plus:
            f.write(_fld("", 80))
        for s in spr:
            f.write(_fld(s, 8))
        if use_plus:
            f.write(_fld(ann_spr, 8))
        for _ in range(ns):
            f.write(_fld("", 32))

        # ---- data records ----
        for r in range(n_records):
            for d, s in zip(digitals, spr):
                f.write(d[r * s:(r + 1) * s].tobytes())
            if use_plus:
                buf = bytes(ann_records[r])
                buf = buf + b"\x00" * (ann_spr * 2 - len(buf))
                f.write(buf)
    return out_path


def _clean(s: str) -> str:
    return "".join(ch if 32 <= ord(ch) < 127 and ch != " " else "_"
                   for ch in str(s)) if s else ""


def _tal(onset: float, duration, text) -> bytes:
    sign = "+" if onset >= 0 else "-"
    s = f"{sign}{abs(onset):g}".encode("ascii")
    if duration is not None and text is not None:
        s += b"\x15" + f"{duration:g}".encode("ascii")
    s += b"\x14"
    if text is not None:
        s += str(text).encode("ascii", "replace")
    s += b"\x14\x00"
    return s
