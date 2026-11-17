#!/usr/bin/env python
"""Write editable YOLO pre-labels for annotation (pothole = 0 from the pothole model, MTW = 1 from COCO class 3).

  python scripts/prelabel.py --images data/interim/test_b_candidates/clip01 --out data/interim/prelabels/clip01 \\
      --pothole-weights models/pothole/best.pt --mtw-weights yolov8s.pt
Import the result into CVAT / Roboflow as YOLO pre-annotations and CORRECT them by hand (see docs/annotation_guide.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import write_prelabels
from ptr.logging_setup import setup_logging


def build_predictor(pothole_w, mtw_w, pc: dict):
    from ultralytics import YOLO  # lazy: only needed for real weights

    pm = YOLO(pothole_w) if pothole_w else None
    mm = YOLO(mtw_w) if mtw_w else None

    def predict(path):
        out = []
        if pm is not None:
            r = pm.predict(str(path), conf=pc["pothole_conf"], verbose=False)[0]
            out += [(pc["pothole_class"], float(c), tuple(map(float, b))) for b, c in
                    zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        if mm is not None:
            r = mm.predict(str(path), conf=pc["mtw_conf"], classes=[pc["mtw_coco_class_id"]], verbose=False)[0]
            out += [(pc["mtw_class"], float(c), tuple(map(float, b))) for b, c in
                    zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        return out

    return predict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pothole-weights", default=None)
    ap.add_argument("--mtw-weights", default=None)
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    setup_logging()
    if not (a.pothole_weights or a.mtw_weights):
        print("give at least one of --pothole-weights / --mtw-weights", file=sys.stderr)
        return 2
    cfg = load_yaml(a.data_prep_config)
    rep = write_prelabels(Path(a.images), Path(a.out), build_predictor(a.pothole_weights, a.mtw_weights, cfg["prelabel"]),
                          cfg["clean"]["image_extensions"])
    print(f"pre-labelled {rep['images']} images, {rep['boxes']} boxes -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
