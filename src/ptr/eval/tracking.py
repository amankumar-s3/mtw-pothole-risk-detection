"""Tracking metrics (ID switches, IDF1, MOTA, MOTP) with motmetrics.

Note: motmetrics 1.4.0's own ``iou_matrix`` uses ``np.asfarray`` (removed in NumPy 2), so the IoU distance
matrix is computed here and passed to the accumulator.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger("eval.tracking")


def load_mot_gt(path: Path) -> pd.DataFrame:
    """MOT 1.1 text: frame,id,x,y,w,h,conf,class,visibility (frames are 1-based in CVAT exports)."""
    df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3, 4, 5], names=["frame", "id", "x", "y", "w", "h"],
                     engine="python", sep=r"\s*,\s*")
    return df.astype({"frame": int, "id": int})


def load_tracks_csv(path: Path, frame_stride: int = 1, frame_offset: int = 1) -> pd.DataFrame:
    """tracks.csv (raw tracker boxes) -> MOT-style frame numbers: gt_frame = frame * stride + offset."""
    t = pd.read_csv(path)
    out = pd.DataFrame({"frame": t["frame"].astype(int) * frame_stride + frame_offset, "id": t["track_id"].astype(int),
                        "x": t["x1"], "y": t["y1"], "w": t["x2"] - t["x1"], "h": t["y2"] - t["y1"]})
    return out


def iou_distance(a: np.ndarray, b: np.ndarray, max_dist: float) -> np.ndarray:
    """1 - IoU for xywh boxes; NaN where the distance exceeds ``max_dist`` (= no match allowed)."""
    d = np.full((len(a), len(b)), np.nan)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            iw = max(0.0, min(x[0] + x[2], y[0] + y[2]) - max(x[0], y[0]))
            ih = max(0.0, min(x[1] + x[3], y[1] + y[3]) - max(x[1], y[1]))
            inter = iw * ih
            union = x[2] * x[3] + y[2] * y[3] - inter
            v = 1.0 - inter / union if union > 0 else 1.0
            if v <= max_dist:
                d[i, j] = v
    return d


def evaluate_tracking(gt: pd.DataFrame, hyp: pd.DataFrame, iou_threshold: float,
                      eval_frames: Optional[set[int]] = None) -> dict:
    import motmetrics as mm

    frames = sorted(eval_frames if eval_frames is not None else set(gt["frame"]) | set(hyp["frame"]))
    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        g, h = gt[gt["frame"] == f], hyp[hyp["frame"] == f]
        d = iou_distance(g[["x", "y", "w", "h"]].to_numpy(float), h[["x", "y", "w", "h"]].to_numpy(float),
                         1.0 - iou_threshold)
        acc.update(g["id"].tolist(), h["id"].tolist(), d, frameid=int(f))
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["num_frames", "num_objects", "idf1", "mota", "motp", "num_switches", "num_misses",
                                 "num_false_positives", "mostly_tracked", "mostly_lost"], name="run").iloc[0]
    motp = float(s["motp"])
    return {"num_frames": int(s["num_frames"]), "num_gt_boxes": int(s["num_objects"]),
            "n_gt_tracks": int(gt["id"].nunique()), "n_hyp_tracks": int(hyp["id"].nunique()),
            "id_switches": int(s["num_switches"]), "idf1": float(s["idf1"]), "mota": float(s["mota"]),
            "motp_distance": motp, "motp_mean_iou": (1.0 - motp) if math.isfinite(motp) else math.nan,
            "misses": int(s["num_misses"]), "false_positives": int(s["num_false_positives"]),
            "mostly_tracked": int(s["mostly_tracked"]), "mostly_lost": int(s["mostly_lost"]),
            "iou_threshold": iou_threshold}
