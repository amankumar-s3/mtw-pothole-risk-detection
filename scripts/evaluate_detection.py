#!/usr/bin/env python
"""Evaluate a detector on named splits (P, R, F1, AP@0.5, AP@0.5:0.95), PR/F1 curves, failure gallery.

Pothole model (confidence chosen on VAL only, then applied to the other splits):
  python scripts/evaluate_detection.py --name pothole_s640 --weights models/pothole/best.pt --target pothole \\
     --split val=data/processed/pothole/val --split test_a=data/processed/pothole/test_a \\
     --split test_b=data/processed/test_b --gt-class 0 --conf-from val --gallery-split test_b
(the Test B pothole class is 0 as well.) COCO motorcycle model on the MTW class of Test B (class 1):
  python scripts/evaluate_detection.py --name mtw_coco --weights yolov8s.pt --target mtw \\
     --split test_b=data/processed/test_b --gt-class 1 --conf 0.30
An IDD motorcycle folder converted to YOLO (motorcycle = class 0) works the same with --gt-class 0.
Needs `pip install ultralytics`. Output: eval_results/detection_<name>.{csv,md}, curves, gallery.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ptr.config import load_pipeline_config, load_yaml
from ptr.eval.detection import run_evaluation
from ptr.logging_setup import setup_logging
from ptr.utils import seed_everything


def make_predictor(weights: str, target: str, pcfg, floor: float):
    from ultralytics import YOLO

    model = YOLO(weights)
    if target == "pothole":
        d = pcfg.detectors.pothole
        kw = dict(imgsz=d.imgsz, iou=d.iou, max_det=d.max_det, device=d.device)
    else:
        d = pcfg.detectors.mtw
        kw = dict(imgsz=d.imgsz, iou=d.iou, device=d.device, classes=[d.coco_class_id])

    def predict(path):
        r = model.predict(str(path), conf=floor, verbose=False, **kw)[0]
        return [(float(c), tuple(map(float, b))) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return predict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--target", choices=["pothole", "mtw"], required=True)
    ap.add_argument("--split", action="append", required=True, metavar="NAME=DIR", help="repeatable; DIR has images/ and labels/")
    ap.add_argument("--gt-class", type=int, required=True, help="class id of the target in the ground-truth labels")
    ap.add_argument("--conf-from", default="val", help="split used to choose the confidence (VAL only)")
    ap.add_argument("--conf", type=float, default=None, help="fixed confidence (e.g. for the COCO MTW model)")
    ap.add_argument("--gallery-split", default=None)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    setup_logging()
    seed_everything(a.seed)
    splits = {}
    for s in a.split:
        if "=" not in s:
            print(f"--split expects NAME=DIR, got {s}", file=sys.stderr)
            return 2
        n, d = s.split("=", 1)
        if not (Path(d) / "images").is_dir():
            print(f"{d}/images not found", file=sys.stderr)
            return 2
        splits[n] = Path(d)
    dp = load_yaml(a.data_prep_config)
    pcfg = load_pipeline_config(a.config)
    predict = make_predictor(a.weights, a.target, pcfg, dp["detection_eval"]["pred_conf_floor"])
    out = run_evaluation(a.name, splits, predict, a.gt_class, dp["clean"]["image_extensions"], dp["detection_eval"],
                         Path(a.out_dir), a.conf_from, a.conf, a.gallery_split)
    print(out["note"])
    for s, r in out["results"].items():
        print(f"{s}: images={r['n_images']} boxes={r['n_gt_boxes']} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} AP50={r['ap50']:.3f} AP50-95={r['ap50_95']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
