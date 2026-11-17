#!/usr/bin/env python
"""Contact sheet of N random annotated images with boxes drawn (for the 50-image visual audit). Seeded.

  python scripts/preview_annotations.py --root data/raw/rdd2022_india --format voc --out docs/audits/rdd2022_india_preview.png
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from ptr.config import load_yaml
from ptr.dataprep import load_annotations


def make_sheet(items, n: int, seed: int, cols: int = 5, cell=(256, 192), only_label: str | None = None) -> np.ndarray:
    rng = random.Random(seed)
    pool = [it for it in items if not only_label or any(b[0] == only_label for b in it[1])]
    pick = rng.sample(pool, min(n, len(pool)))
    rows = max(1, -(-len(pick) // cols))
    sheet = np.zeros((rows * cell[1], cols * cell[0], 3), np.uint8)
    for k, (ip, boxes, (w, h)) in enumerate(pick):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        for lab, x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), max(2, w // 300))
            cv2.putText(img, str(lab), (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, w / 1200), (0, 255, 0), max(1, w // 600))
        img = cv2.resize(img, cell)
        cv2.putText(img, ip.name[:34], (3, cell[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        r, c = divmod(k, cols)
        sheet[r * cell[1]:(r + 1) * cell[1], c * cell[0]:(c + 1) * cell[0]] = img
    return sheet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--format", choices=["voc", "yolo"], required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-label", default=None, help="only images containing this class label (e.g. D40)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    exts = load_yaml(a.data_prep_config)["clean"]["image_extensions"]
    items = load_annotations(Path(a.root), a.format, exts)
    if not items:
        print("no annotated images found", file=sys.stderr)
        return 2
    sheet = make_sheet(items, a.n, a.seed, only_label=a.only_label)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f"wrote {a.out} ({min(a.n, len(items))} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
