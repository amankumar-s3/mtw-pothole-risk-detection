#!/usr/bin/env python
"""Train the single-class pothole detector (YOLOv8, COCO-pretrained start). Intended for Google Colab.

  python scripts/train_pothole.py --config configs/train_pothole.yaml
  python scripts/train_pothole.py --config configs/train_pothole.yaml --resume      # after a disconnect
  python scripts/train_pothole.py --config configs/train_pothole.yaml --dry-run     # print resolved arguments only
Run folder: <project>/<name>/ with weights/last.pt (saved every epoch), weights/best.pt and run_config.yaml.
Vertical flip and rotation are disabled in the config (road geometry is fixed). Nothing is invented: metrics are
whatever Ultralytics computes on the val split.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

import yaml

from ptr.config import load_yaml
from ptr.logging_setup import setup_logging
from ptr.utils import seed_everything


def build_args(cfg: dict) -> dict:
    keys = ["data", "imgsz", "epochs", "patience", "batch", "seed", "workers", "project", "name", "device", "cos_lr"]
    kw = {k: cfg[k] for k in keys if cfg.get(k) is not None}
    kw.update(cfg.get("augment", {}))
    kw["deterministic"] = True
    return kw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_pothole.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    for k in ("model", "data", "project", "name", "device"):
        ap.add_argument(f"--{k}", default=None)
    for k in ("imgsz", "epochs", "patience", "batch", "seed", "workers"):
        ap.add_argument(f"--{k}", type=int, default=None)
    a = ap.parse_args(argv)
    log = setup_logging()
    cfg = load_yaml(a.config)
    for k in ("model", "data", "project", "name", "device", "imgsz", "epochs", "patience", "batch", "seed", "workers"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    kw = build_args(cfg)
    if a.dry_run:
        print(yaml.safe_dump({"model": cfg["model"], "resume": a.resume, "train_args": kw}, sort_keys=False))
        return 0
    seed_everything(int(cfg["seed"]))
    if not Path(cfg["data"]).is_file():
        print(f"dataset yaml not found: {cfg['data']} (run scripts/make_splits.py)", file=sys.stderr)
        return 2
    from ultralytics import YOLO
    import ultralytics

    run_dir = Path(cfg["project"]) / cfg["name"]
    last = run_dir / "weights" / "last.pt"
    if a.resume:
        if not last.is_file():
            print(f"nothing to resume: {last} not found", file=sys.stderr)
            return 2
        import torch

        ck = torch.load(str(last), map_location="cpu", weights_only=False)      # our own checkpoint
        if ck.get("optimizer") is None or ck.get("epoch", -1) == -1 or not ck.get("train_args"):
            print(f"{last} is a finished (stripped) checkpoint: nothing to resume. "
                  f"Start a new run (different --name) or use best.pt.", file=sys.stderr)
            return 0
        log.info("resuming from %s (epoch %s)", last, ck.get("epoch"))
        YOLO(str(last)).train(resume=True)
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    data_text = Path(cfg["data"]).read_text()
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump({
        "train_config": cfg, "ultralytics": ultralytics.__version__, "python": platform.python_version(),
        "dataset_yaml_sha256": hashlib.sha256(data_text.encode()).hexdigest(), "dataset_yaml": data_text}, sort_keys=False))
    log.info("training %s with %s", cfg["model"], kw)
    YOLO(cfg["model"]).train(exist_ok=True, **kw)
    print(f"done. weights: {run_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
