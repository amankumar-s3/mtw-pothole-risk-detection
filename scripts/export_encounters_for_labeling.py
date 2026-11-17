#!/usr/bin/env python
"""Export a short clip per event for BLIND human labelling, plus a blank CSV for the raters.

  python scripts/export_encounters_for_labeling.py --run outputs/<run_id> [--run outputs/<other>] --out labeling
Clips are cut from the ORIGINAL video and carry NO risk class, score or reason. By default two thin neutral boxes mark
the MTW and the pothole of the event (no colour coding by risk; use --no-mark for completely clean clips).
CSV columns: event_id, clip_id, rater1, rater2, notes. Raters must not see events.csv or the system output.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

from ptr.config import load_yaml
from ptr.logging_setup import setup_logging


def export_clip(video: Path, out_path: Path, start_t: float, end_t: float, fps: float, boxes_for_frame, scale: float,
                mark: bool) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_t * fps)))
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n, idx = 0, max(0, int(start_t * fps))
    while idx <= int(end_t * fps):
        ok, f = cap.read()
        if not ok:
            break
        if mark:
            for label, b in boxes_for_frame(idx):
                x1, y1, x2, y2 = [int(v / scale) for v in b]
                cv2.rectangle(f, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(f, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        wr.write(f)
        n += 1
        idx += 1
    wr.release()
    cap.release()
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="labeling")
    ap.add_argument("--no-mark", action="store_true")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    setup_logging()
    pad = load_yaml(a.data_prep_config)["labeling"]["clip_pad_seconds"]
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows = []
    for run in map(Path, a.run):
        s = json.loads((run / "summary.json").read_text())
        ev = pd.read_csv(run / "events.csv")
        tr = pd.read_csv(run / "tracks.csv")
        po = pd.read_csv(run / "potholes.csv")
        stride = round(s["video_fps"] / s["effective_fps"])
        cap = cv2.VideoCapture(s["video"])
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        scale = s["frame_size"][0] / vw if vw else 1.0
        for e in ev.to_dict("records"):
            t0, t1 = max(0.0, e["start_t"] - pad), e["end_t"] + pad

            def boxes(vidx, e=e):
                k = vidx // stride
                out_b = []
                a_ = tr[(tr["track_id"] == e["mtw_id"]) & (tr["frame"] == k)]
                if len(a_):
                    r = a_.iloc[0]
                    out_b.append(("MTW", (r.x1, r.y1, r.x2, r.y2)))
                b_ = po[(po["pothole_id"] == e["pothole_id"]) & (po["frame"] == k)]
                if len(b_):
                    r = b_.iloc[0]
                    out_b.append(("pothole", (r.x1, r.y1, r.x2, r.y2)))
                return out_b

            name = f"{s['clip_id']}__{e['event_id']}.mp4"
            export_clip(Path(s["video"]), out / "clips" / name, t0, t1, s["video_fps"], boxes, scale, not a.no_mark)
            rows.append({"event_id": e["event_id"], "clip_id": s["clip_id"], "rater1": "", "rater2": "", "notes": ""})
    with open(out / "labels_template.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["event_id", "clip_id", "rater1", "rater2", "notes"])
        w.writeheader()
        w.writerows(rows)
    print(f"exported {len(rows)} clips to {out / 'clips'} and {out / 'labels_template.csv'} "
          f"(labels allowed: Low / Moderate / High)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
