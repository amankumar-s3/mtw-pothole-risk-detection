#!/usr/bin/env python
"""Split events into a CALIBRATION set and a HELD-OUT set BY CLIP (never by event).

  python scripts/split_events_by_clip.py --events outputs/runA/events.csv outputs/runB/events.csv --out eval_results
Calibrate thresholds on the calibration clips only; report agreement on the held-out clips. The number of clips,
events (and raters) behind each metric must be stated in the report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ptr.config import load_yaml
from ptr.eval.risk_agreement import split_by_clip
from ptr.logging_setup import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    setup_logging()
    lab = load_yaml(a.data_prep_config)["labeling"]
    frac = a.fraction if a.fraction is not None else lab["calibration_fraction"]
    seed = a.seed if a.seed is not None else lab["seed"]
    ev = pd.concat([pd.read_csv(p) for p in a.events], ignore_index=True)
    if "clip_id" not in ev.columns:
        print("events.csv has no clip_id column", file=sys.stderr)
        return 2
    cal, held = split_by_clip(ev["clip_id"].tolist(), frac, seed)
    assert not set(cal) & set(held)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev[ev["clip_id"].isin(cal)].to_csv(out / "events_calibration.csv", index=False)
    ev[ev["clip_id"].isin(held)].to_csv(out / "events_heldout.csv", index=False)
    msg = [f"# Event split by clip", "", f"- calibration clips ({len(cal)}): {cal}", f"- held-out clips ({len(held)}): {held}",
           f"- events: calibration {int(ev['clip_id'].isin(cal).sum())}, held-out {int(ev['clip_id'].isin(held).sum())}",
           "- no clip appears in both sets"]
    (out / "event_split.md").write_text("\n".join(msg) + "\n")
    print("\n".join(msg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
