#!/usr/bin/env python
"""Run the full pipeline on one video.

  python scripts/run_pipeline.py --video clip.mp4 --config configs/pipeline.yaml --risk-config configs/risk.yaml
  python scripts/run_pipeline.py --video data/videos/synthetic_demo.mp4 --mock-detections data/videos/synthetic_demo.json
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from ptr.config import load_pipeline_config, load_risk_config
from ptr.detection.detector import MockDetections, RealSource
from ptr.pipeline import Pipeline
from ptr.utils import seed_everything


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--risk-config", default="configs/risk.yaml")
    ap.add_argument("--out", default="outputs", help="parent folder; a run folder <run_id> is created inside")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--mock-detections", default=None, help="JSON file replacing both detectors (no weights needed)")
    ap.add_argument("--camera-mode", choices=["moving", "static"], default=None, help="override camera_mode")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="skip annotated.mp4")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    seed_everything(a.seed)
    cfg = load_pipeline_config(a.config)
    risk_cfg = load_risk_config(a.risk_config)
    if a.camera_mode:
        cfg = dataclasses.replace(cfg, camera_mode=a.camera_mode)
    if a.max_frames is not None:
        cfg = dataclasses.replace(cfg, video=dataclasses.replace(cfg.video, max_frames=a.max_frames))
    if a.no_video:
        cfg = dataclasses.replace(cfg, output=dataclasses.replace(cfg.output, write_video=False))
    if a.mock_detections:
        source = MockDetections.from_file(a.mock_detections, cfg.detectors.pothole.min_box_px,
                                          cfg.detectors.mtw.min_box_px)
    else:
        source = RealSource(cfg)
    pipe = Pipeline(cfg, risk_cfg, source, out_root=a.out, run_id=a.run_id,
                    meta={"mock_detections": a.mock_detections, "seed": a.seed,
                          "config_file": a.config, "risk_config_file": a.risk_config})
    summary = pipe.run(a.video)
    print(f"run folder: {pipe.run_dir}")
    print(f"events per class: {summary['events_per_class']}  (potential-risk indicator, not accident prediction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
