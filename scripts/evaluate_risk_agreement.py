#!/usr/bin/env python
"""System-vs-human risk agreement (confusion matrix, accuracy, weighted kappa, per-class P/R) and inter-rater kappa.

  python scripts/evaluate_risk_agreement.py --events outputs/<run>/events.csv --labels labels_heldout.csv
The labels CSV (from export_encounters_for_labeling.py) has columns event_id, clip_id, rater1, rater2, notes
[, consensus]. The reference is `consensus` if present, else rater1. The sample size is always printed.
This is agreement with human judgement, not crash validity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ptr.eval.risk_agreement import agreement_report, inter_rater, load_labelled_events
from ptr.logging_setup import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True, nargs="+", help="one or more events.csv")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--name", default="risk")
    ap.add_argument("--out-dir", default="eval_results")
    a = ap.parse_args(argv)
    setup_logging()
    events = pd.concat([pd.read_csv(p) for p in a.events], ignore_index=True)
    tmp = Path(a.out_dir) / f"_events_{a.name}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(tmp, index=False)
    df = load_labelled_events(tmp, Path(a.labels))
    tmp.unlink()
    if df.empty:
        print("no events with complete human labels", file=sys.stderr)
        return 2
    ref_col = "consensus" if "consensus" in df.columns and df["consensus"].notna().all() else "rater1"
    rep = agreement_report(df[ref_col].tolist(), df["peak_class"].tolist())
    rep["reference"] = ref_col
    rep["n_clips"] = int(df["clip_id"].nunique()) if "clip_id" in df.columns else None
    if "rater2" in df.columns:
        rep["inter_rater"] = inter_rater(df["rater1"].tolist(), df["rater2"].tolist())
    out = Path(a.out_dir)
    (out / f"risk_agreement_{a.name}.json").write_text(json.dumps(rep, indent=2))
    cm = rep["confusion_reference_rows_system_cols"]
    L = [f"# Risk agreement: {a.name}", "", f"- events with complete labels (n): **{rep['n']}**, clips: {rep['n_clips']}",
         f"- reference labels: `{ref_col}`; system label = event peak class", f"- accuracy: {rep['accuracy']:.3f}",
         f"- weighted Cohen's kappa (linear): {rep['weighted_kappa']:.3f}", "",
         "| human \\ system | Low | Moderate | High |", "|---|---|---|---|"]
    for lab, row in zip(["Low", "Moderate", "High"], cm):
        L.append(f"| {lab} | {row[0]} | {row[1]} | {row[2]} |")
    L += ["", "| class | precision | recall | n human | n system |", "|---|---|---|---|---|"]
    for lab, d in rep["per_class"].items():
        L.append(f"| {lab} | {d['precision']:.3f} | {d['recall']:.3f} | {d['n_reference']} | {d['n_system']} |")
    if "inter_rater" in rep:
        ir = rep["inter_rater"]
        L += ["", f"- inter-rater weighted kappa: {ir['weighted_kappa']:.3f}, exact agreement {ir['exact_agreement']:.3f} (n={ir['n']})"]
    if rep["n"] < 30:
        L += ["", "WARNING: fewer than ~30 labelled events: report as an exploratory case study (docs/limitations.md)."]
    (out / f"risk_agreement_{a.name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
