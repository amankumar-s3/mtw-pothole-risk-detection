#!/usr/bin/env python
"""ID switches, IDF1, MOTA, MOTP of a run against MOT-format ground truth (CVAT 'MOT 1.1', see docs/mot_annotation_guide.md).

  python scripts/evaluate_tracking.py --gt data/mot_gt/clip01/gt.txt --run outputs/<run_id>
Frames: gt_frame = processed_frame * frame_stride + 1 (CVAT is 1-based). Only frames the pipeline processed are scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ptr.config import load_yaml
from ptr.eval.tracking import evaluate_tracking, load_mot_gt, load_tracks_csv
from ptr.logging_setup import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--run", required=True, help="run folder with tracks.csv, egomotion.csv, summary.json")
    ap.add_argument("--frame-offset", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    setup_logging()
    run = Path(a.run)
    summary = json.loads((run / "summary.json").read_text())
    stride = round(summary["video_fps"] / summary["effective_fps"])
    gt = load_mot_gt(Path(a.gt))
    hyp = load_tracks_csv(run / "tracks.csv", stride, a.frame_offset)
    processed = set((pd.read_csv(run / "egomotion.csv")["frame"].astype(int) * stride + a.frame_offset).tolist())
    thr = load_yaml(a.data_prep_config)["tracking_eval"]["iou_threshold"]
    res = evaluate_tracking(gt[gt["frame"].isin(processed)], hyp, thr, processed)
    name = a.name or run.name
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"tracking_{name}.json").write_text(json.dumps(res, indent=2))
    lines = [f"# Tracking evaluation: {name}", "", f"- clip: {summary['clip_id']}, frames scored: {res['num_frames']}, "
             f"GT boxes: {res['num_gt_boxes']}, GT tracks: {res['n_gt_tracks']}, system tracks: {res['n_hyp_tracks']}",
             f"- IoU threshold {thr}", "", "| ID switches | IDF1 | MOTA | MOTP (mean IoU of matches) | misses | false positives |",
             "|---|---|---|---|---|---|",
             f"| {res['id_switches']} | {res['idf1']:.3f} | {res['mota']:.3f} | {res['motp_mean_iou']:.3f} | "
             f"{res['misses']} | {res['false_positives']} |", "",
             "Short hand-annotated clips: state their length and track count wherever these numbers are quoted."]
    (out / f"tracking_{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
