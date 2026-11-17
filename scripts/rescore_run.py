#!/usr/bin/env python
"""Re-score a finished run with a (modified) risk.yaml WITHOUT re-processing the video.

  python scripts/rescore_run.py --run outputs/<run_id> --risk-config configs/risk.yaml --out outputs/<run_id>/events_rescored.csv
Uses the saved interactions.csv (features do not depend on the risk config), so editing weights, anchors, class cuts,
gates or persistence in risk.yaml is a seconds-long calibration loop. Event ids stay the same as long as the features
are unchanged, so existing human labels (keyed by event_id + clip_id) remain valid. Changing pipeline.yaml (detectors,
features, registry, ego-motion) requires re-running run_pipeline.py and new labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ptr.analysis.events import EventAggregator
from ptr.analysis.rescore import rescore
from ptr.config import load_risk_config
from ptr.logging_setup import setup_logging
from ptr.pipeline import EVENT_COLS
from ptr.utils import write_csv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--risk-config", default="configs/risk.yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    setup_logging()
    run = Path(a.run)
    s = json.loads((run / "summary.json").read_text())
    risk = load_risk_config(a.risk_config)
    df = pd.read_csv(run / "interactions.csv")
    events = rescore(df, risk.for_mode(s["camera_mode"]), s["effective_fps"])
    rows = []
    for e in events:
        d = e.to_row()
        d["clip_id"] = s["clip_id"]
        rows.append(d)
    out = Path(a.out) if a.out else run / "events_rescored.csv"
    write_csv(out, rows, EVENT_COLS)
    summ = EventAggregator.summarize(events)
    print(f"{out}: {summ['n_events']} events {summ['events_per_class']} (camera mode {s['camera_mode']}, {risk.status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
