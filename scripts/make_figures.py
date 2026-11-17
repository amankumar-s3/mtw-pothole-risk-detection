#!/usr/bin/env python
"""Figures from a run folder: risk-vs-time per MTW id, class-count bars, pothole position scatter.

  python scripts/make_figures.py --run outputs/<run_id>        -> outputs/<run_id>/figures/*.png
The pothole scatter is only meaningful for a static camera (in a moving camera it shows image positions, not a map).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ptr.logging_setup import setup_logging
from ptr.schemas import RISK_LABELS

COL = {"Low": "#46be46", "Moderate": "#ffa500", "High": "#e62828"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    setup_logging()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((run / "summary.json").read_text())
    inter = pd.read_csv(run / "interactions.csv")
    ev = pd.read_csv(run / "events.csv")
    po = pd.read_csv(run / "potholes.csv")
    written = []

    ids = sorted(inter["mtw_id"].unique()) if len(inter) else []
    fig, axes = plt.subplots(max(1, len(ids)), 1, figsize=(8, 2.2 * max(1, len(ids))), sharex=True, squeeze=False)
    for ax, mid in zip(axes[:, 0], ids or [None]):
        if mid is None:
            ax.text(0.5, 0.5, "no MTW-pothole interactions", ha="center")
            continue
        d = inter[inter["mtw_id"] == mid].sort_values("t")
        top = d.groupby("t").agg(total=("total", "max")).reset_index()
        ax.plot(top["t"], top["total"], color="0.6", lw=1)
        for lab in RISK_LABELS:
            m = d[d["accepted_label"] == lab].groupby("t")["total"].max()
            ax.scatter(m.index, m.values, s=8, color=COL[lab], label=lab)
        ax.set_ylabel(f"MTW {mid}\nscore")
        ax.set_ylim(0, 1.05)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(ncol=3, fontsize=7, loc="upper right")
    axes[0, 0].set_title(f"risk score over time ({s['clip_id']}; {s['risk_status']}; potential-risk indicator)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "risk_timeline.png", dpi=120)
    plt.close(fig)
    written.append("risk_timeline.png")

    counts = s["events_per_class"]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(RISK_LABELS, [counts[l] for l in RISK_LABELS], color=[COL[l] for l in RISK_LABELS])
    ax.set_ylabel("events")
    ax.set_title(f"events per peak class (n={sum(counts.values())})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "class_counts.png", dpi=120)
    plt.close(fig)
    written.append("class_counts.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    c = po[po["confirmed"] == 1]
    if len(c):
        ax.scatter((c["x1"] + c["x2"]) / 2, (c["y1"] + c["y2"]) / 2, s=6, c=c["pothole_id"], cmap="tab20")
    ax.set_xlim(0, s["frame_size"][0])
    ax.set_ylim(s["frame_size"][1], 0)
    ax.set_title("confirmed pothole centres in image coordinates\n(a map only for a static camera)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pothole_positions.png", dpi=120)
    plt.close(fig)
    written.append("pothole_positions.png")
    print(f"wrote {written} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
