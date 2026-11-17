#!/usr/bin/env python
"""Latency / FPS of each stage on THIS machine (warm-up excluded, hardware logged). Informational only.

  python scripts/benchmark_fps.py --video data/videos/clip01.mp4 --frames 120 --warmup 10 \\
      --pothole-weights models/pothole/best.pt --mtw-weights yolov8s.pt
The numbers describe offline processing on the stated hardware; they are not a real-time guarantee.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2

from ptr.config import load_pipeline_config
from ptr.eval.fps import benchmark, hardware_info
from ptr.logging_setup import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--pothole-weights", default=None)
    ap.add_argument("--mtw-weights", default=None)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--name", default="fps")
    ap.add_argument("--out-dir", default="eval_results")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_pipeline_config(a.config)
    cap = cv2.VideoCapture(a.video)
    frames = []
    while len(frames) < a.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) <= a.warmup:
        print("video too short for this warm-up", file=sys.stderr)
        return 2
    det = cfg.detectors
    if a.pothole_weights:
        det = dataclasses.replace(det, pothole=dataclasses.replace(det.pothole, weights=a.pothole_weights))
    if a.mtw_weights:
        det = dataclasses.replace(det, mtw=dataclasses.replace(det.mtw, weights=a.mtw_weights))
    stages = {}
    if a.pothole_weights:
        from ptr.detection.detector import PotholeDetector
        pd_ = PotholeDetector(det.pothole)
        stages["pothole_detector"] = pd_.detect
    if a.mtw_weights:
        from ptr.tracking.mtw_tracker import MTWTracker
        mt = MTWTracker(det.mtw)
        stages["mtw_detector_bytetrack"] = mt.track
    if len(stages) == 2:
        both = (stages["pothole_detector"], stages["mtw_detector_bytetrack"])
        stages = {**stages, "both_detectors": lambda f: (both[0](f), both[1](f))}
    if not stages:
        print("give --pothole-weights and/or --mtw-weights", file=sys.stderr)
        return 2
    hw = hardware_info()
    rows = {}
    for name, fn in stages.items():
        rows[name] = benchmark(fn, frames, a.warmup)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps({"hardware": hw, "frame_size": list(frames[0].shape[1::-1]),
                                                    "stages": rows}, indent=2))
    L = [f"# Speed benchmark: {a.name}", "", f"- hardware: {hw}", f"- frames timed per stage: {a.frames - a.warmup} "
         f"(warm-up {a.warmup} excluded), frame size {frames[0].shape[1]}x{frames[0].shape[0]}", "",
         "| stage | mean ms | median ms | p95 ms | FPS (mean) |", "|---|---|---|---|---|"]
    for n, r in rows.items():
        L.append(f"| {n} | {r['mean_ms']:.1f} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | {r['fps_mean']:.1f} |")
    L += ["", "Detector time only (no ego-motion, features or drawing). Offline processing speed, not a real-time claim."]
    (out / f"{a.name}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
