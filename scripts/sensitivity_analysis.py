#!/usr/bin/env python
"""Perturb risk weights and thresholds by +/-20% (config: sensitivity.perturbation) and count class changes.

  python scripts/sensitivity_analysis.py --run outputs/<run_id> [--run outputs/<other>] --risk-config configs/risk.yaml
Re-scores the saved interactions.csv (no video needed). The baseline is the unperturbed config re-scored the same way.
The numbers are for the INITIAL-HYPOTHESIS parameters: they show how fragile the labels are, not that they are right.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

from ptr.analysis.rescore import rescore
from ptr.config import load_risk_config, load_yaml
from ptr.logging_setup import setup_logging
from ptr.schemas import RISK_LABELS


def perturbations(block, p: float):
    """Yield (name, factor, modified block): one parameter at a time, x(1-p) and x(1+p)."""
    def rep(obj, **kw):
        return dataclasses.replace(obj, **kw)
    for f in (1 - p, 1 + p):
        for k in ("proximity", "ttr", "swerve", "size", "decel"):
            w = block.weights
            if getattr(w, k) > 0:
                yield f"weight.{k}", f, rep(block, weights=rep(w, **{k: getattr(w, k) * f}))
        for k in ("proximity", "ttr", "swerve", "decel"):
            a = getattr(block.anchors, k)
            yield f"anchor.{k}.zero_at", f, rep(block, anchors=rep(block.anchors, **{k: rep(a, zero_at=a.zero_at * f)}))
            yield f"anchor.{k}.one_at", f, rep(block, anchors=rep(block.anchors, **{k: rep(a, one_at=a.one_at * f)}))
        for k in ("low_max", "high_min"):
            yield f"classes.{k}", f, rep(block, classes=rep(block.classes, **{k: getattr(block.classes, k) * f}))
        for k in ("p_min", "s_min", "b_min", "z_min"):
            yield f"gate.{k}", f, rep(block, high_gate=rep(block.high_gate, **{k: getattr(block.high_gate, k) * f}))


def peak_by_pair(events):
    """(mtw, pothole) -> list of (start, end, peak rank)."""
    d = {}
    for e in events:
        d.setdefault((e.mtw_id, e.pothole_id), []).append((e.start_frame, e.end_frame, RISK_LABELS.index(e.peak_class)))
    return d


def compare(base_events, pert_events):
    pb = peak_by_pair(pert_events)
    up = down = lost = 0
    for e in base_events:
        ranks = [r for (s, t, r) in pb.get((e.mtw_id, e.pothole_id), []) if s <= e.end_frame and t >= e.start_frame]
        if not ranks:
            lost += 1
            continue
        r = max(ranks)
        b = RISK_LABELS.index(e.peak_class)
        up += r > b
        down += r < b
    return up, down, lost


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--risk-config", default="configs/risk.yaml")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    ap.add_argument("--name", default="sensitivity")
    ap.add_argument("--out-dir", default="eval_results")
    a = ap.parse_args(argv)
    setup_logging()
    risk = load_risk_config(a.risk_config)
    p = load_yaml(a.data_prep_config)["sensitivity"]["perturbation"]
    runs = []
    for r in a.run:
        s = json.loads((Path(r) / "summary.json").read_text())
        df = pd.read_csv(Path(r) / "interactions.csv")
        runs.append((s, df))
    modes = {s["camera_mode"] for s, _ in runs}
    if len(modes) != 1:
        print(f"runs mix camera modes {modes}; analyse them separately", file=sys.stderr)
        return 2
    mode = modes.pop()
    block = risk.for_mode(mode)
    base = [rescore(df, block, s["effective_fps"]) for s, df in runs]
    n_base = sum(len(b) for b in base)
    rows = []
    for name, f, blk in perturbations(block, p):
        up = down = lost = 0
        for (s, df), be in zip(runs, base):
            u, d, l = compare(be, rescore(df, blk, s["effective_fps"]))
            up, down, lost = up + u, down + d, lost + l
        rows.append({"parameter": name, "factor": round(f, 3), "events": n_base, "class_up": up, "class_down": down,
                     "event_lost": lost, "changed": up + down + lost})
    out = pd.DataFrame(rows)
    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    out.to_csv(outd / f"{a.name}.csv", index=False)
    top = out.sort_values("changed", ascending=False).head(10)
    L = [f"# Sensitivity analysis ({a.name})", "", f"- status of the parameters: {risk.status}", f"- camera mode block: {mode}",
         f"- runs: {len(runs)}, baseline events: {n_base}, perturbation +/-{p * 100:.0f}% one parameter at a time",
         "- `class_up/down`: peak class changed; `event_lost`: no overlapping event after the change", "",
         "| parameter | factor | up | down | lost | changed |", "|---|---|---|---|---|---|"]
    for r in top.to_dict("records"):
        L.append(f"| {r['parameter']} | {r['factor']} | {r['class_up']} | {r['class_down']} | {r['event_lost']} | {r['changed']} |")
    L += ["", f"Full table: `{a.name}.csv` ({len(out)} perturbations). With few events these counts are exploratory."]
    (outd / f"{a.name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
