"""
convert.py
==========
Convert DSI Ponemah experiments to EDF/EDF+ (pure Python; no DSI software needed)
and provide a command-line interface, including batch conversion.

Telemetry recordings can be discontinuous. Two strategies are offered:

  per_segment (default, lossless): one EDF per contiguous acquisition segment,
      each carrying that segment's true start time.
  single: one concatenated EDF with an EDF+ annotation at each gap recording its
      real length (on-disk timeline is compressed; gaps documented).

ECG channels are written so that the stored 16-bit digital values equal the
original Ponemah ADC counts, with the physical range set to the validated
+/-10 mV span -- i.e. an essentially lossless, exactly-calibrated round trip.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import timedelta
from typing import Callable, List, Optional

import numpy as np

from edf_writer import Channel, write_edf
from ponemah_reader import (
    MockPonemahReader, PonemahReader, PurePythonPonemahReader, SubjectInfo,
    find_experiment_files,
)

# Ponemah ADC digital range (validated from CHANNEL config).
DIG_MIN, DIG_MAX = -32256, 32255


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s)).strip("_") or "out"


def _gather_segment(reader: PonemahReader, channels, segment) -> List[Channel]:
    out = []
    for ch in channels:
        if ch.sample_rate <= 0:
            continue
        samples = np.concatenate(list(reader.read_channel_segment(ch.channel_id, segment)))
        scale = getattr(ch, "scale", 1.0)
        if getattr(ch, "digital", False):
            # Lossless: store original ADC counts; physical range carries the calibration.
            out.append(Channel(
                label=ch.label, samples=samples, sample_rate=ch.sample_rate,
                physical_dimension=ch.unit,
                physical_min=DIG_MIN * scale, physical_max=DIG_MAX * scale,
                digital=True, digital_min=DIG_MIN, digital_max=DIG_MAX,
            ))
        else:
            out.append(Channel(
                label=ch.label, samples=samples, sample_rate=ch.sample_rate,
                physical_dimension=ch.unit,
            ))
    return out


def convert_subject(reader, subject, out_dir, mode="per_segment",
                    channel_labels=None, progress=None, chunk_seconds=None) -> List[str]:
    log = progress or (lambda m: None)
    channels = subject.channels
    if channel_labels:
        wanted = {c.lower() for c in channel_labels}
        channels = [c for c in channels if c.label.lower() in wanted]
    if not channels:
        log(f"  {subject.name}: no matching channels, skipped")
        return []
    ref = channels[0].segments or []
    if not ref:
        log(f"  {subject.name}: no segments, skipped")
        return []

    written = []
    if mode == "per_segment":
        for i, seg in enumerate(ref):
            edf_channels = _gather_segment(reader, channels, seg)
            if chunk_seconds and chunk_seconds > 0:
                dur = max(len(ec.samples) / ec.sample_rate for ec in edf_channels)
                nchunks = max(1, int(math.ceil(dur / chunk_seconds)))
                for c in range(nchunks):
                    sub = []
                    for ec in edf_channels:
                        a = int(round(c * chunk_seconds * ec.sample_rate))
                        b = int(round((c + 1) * chunk_seconds * ec.sample_rate))
                        part = ec.samples[a:b]
                        if part.size == 0:
                            continue
                        sub.append(Channel(ec.label, part, ec.sample_rate,
                                           ec.physical_dimension, ec.physical_min,
                                           ec.physical_max, ec.prefilter, ec.transducer,
                                           ec.digital, ec.digital_min, ec.digital_max))
                    if not sub:
                        continue
                    cstart = seg.start_utc + timedelta(seconds=c * chunk_seconds)
                    fn = f"{_safe(subject.name)}_seg{i+1:02d}_part{c+1:03d}_{cstart:%Y%m%d_%H%M%S}.edf"
                    write_edf(os.path.join(out_dir, fn), sub, start_time=cstart,
                              patient_code=_safe(subject.name), patient_name=subject.name,
                              recording_additional=f"seg {i+1}/{len(ref)} part {c+1}/{nchunks}")
                    written.append(fn)
                    log(f"  wrote {fn}")
            else:
                stamp = seg.start_utc.strftime("%Y%m%d_%H%M%S")
                fn = f"{_safe(subject.name)}_seg{i+1:02d}_{stamp}.edf"
                write_edf(os.path.join(out_dir, fn), edf_channels, start_time=seg.start_utc,
                          patient_code=_safe(subject.name), patient_name=subject.name,
                          recording_additional=f"segment {i+1}/{len(ref)}")
                written.append(fn)
                log(f"  wrote {fn}")
    elif mode == "single":
        per_ch = {c.label: [] for c in channels}
        annotations, running, prev_end = [], 0.0, None
        for i, seg in enumerate(ref):
            edf_channels = _gather_segment(reader, channels, seg)
            seg_len = min(len(ec.samples) / ec.sample_rate for ec in edf_channels)
            if prev_end is not None:
                gap = (seg.start_utc - prev_end).total_seconds()
                annotations.append((running, 0.0, f"GAP {gap:.1f}s before segment {i+1}"))
            for ec in edf_channels:
                per_ch[ec.label].append(ec)
            running += seg_len
            prev_end = seg.end_utc
        merged = []
        for c in channels:
            parts = per_ch[c.label]
            merged.append(Channel(c.label, np.concatenate([p.samples for p in parts]),
                                  parts[0].sample_rate, parts[0].physical_dimension,
                                  parts[0].physical_min, parts[0].physical_max))
        fn = f"{_safe(subject.name)}_concat.edf"
        path = os.path.join(out_dir, fn)
        write_edf(path, merged, start_time=ref[0].start_utc,
                  patient_code=_safe(subject.name), patient_name=subject.name,
                  recording_additional="concatenated; gaps in annotations",
                  annotations=annotations)
        written.append(path)
        log(f"  wrote {fn}  ({len(annotations)} gap annotations)")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return written


def convert_experiment(exp_folder, out_dir, reader, mode="per_segment",
                       subject_names=None, channel_labels=None, progress=None,
                       chunk_seconds=None):
    log = progress or print
    os.makedirs(out_dir, exist_ok=True)
    subjects = reader.list_subjects()
    if subject_names:
        keep = {s.lower() for s in subject_names}
        subjects = [s for s in subjects if s.name.lower() in keep]
    log(f"Experiment: {exp_folder}  ({len(subjects)} subject(s), mode={mode})")
    written = []
    for sub in subjects:
        log(f"Subject {sub.name}: {len(sub.channels)} channel(s)")
        written += convert_subject(reader, sub, out_dir, mode, channel_labels, progress, chunk_seconds)
    log(f"Done. {len(written)} EDF file(s) in {out_dir}")
    return written


def make_reader(args) -> PonemahReader:
    if args.mock:
        return MockPonemahReader()
    return PurePythonPonemahReader(args.exp_folder, include_aux=not args.ecg_only,
                                   verbose=True, max_seconds=args.max_seconds)


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert DSI Ponemah experiments to EDF/EDF+.")
    p.add_argument("exp_folder", nargs="?", help="Experiment folder (.PnmWav + .PnmExp + .PnmMarks).")
    p.add_argument("-o", "--out-dir", default="edf_out")
    p.add_argument("--mode", choices=["per_segment", "single"], default="per_segment")
    p.add_argument("--subjects", nargs="*")
    p.add_argument("--channels", nargs="*", help="Only these channel labels (e.g. ECG).")
    p.add_argument("--ecg-only", action="store_true", help="Skip non-ECG (aux) channels.")
    p.add_argument("--max-seconds", type=float, default=None, help="Only convert the first N seconds.")
    p.add_argument("--chunk-minutes", type=float, default=None, help="Split output into chunks of N minutes (e.g. 60).")
    p.add_argument("--batch", help="Parent folder; convert every experiment subfolder.")
    p.add_argument("--mock", action="store_true", help="Synthetic demo (no data needed).")
    args = p.parse_args(argv)

    if args.batch:
        subdirs = [os.path.join(args.batch, d) for d in os.listdir(args.batch)
                   if os.path.isdir(os.path.join(args.batch, d))]
        exps = [d for d in subdirs if find_experiment_files(d)[0]]
        print(f"Batch: {len(exps)} experiment folder(s) under {args.batch}")
        for d in exps:
            args.exp_folder = d
            with make_reader(args) as reader:
                convert_experiment(d, os.path.join(args.out_dir, _safe(os.path.basename(d))),
                                   reader, args.mode, args.subjects, args.channels,
                                   chunk_seconds=(args.chunk_minutes*60 if args.chunk_minutes else None))
        return
    if not args.exp_folder:
        p.error("exp_folder is required unless --batch is used.")
    with make_reader(args) as reader:
        convert_experiment(args.exp_folder, args.out_dir, reader,
                           args.mode, args.subjects, args.channels,
                           chunk_seconds=(args.chunk_minutes*60 if args.chunk_minutes else None))


if __name__ == "__main__":
    main()
