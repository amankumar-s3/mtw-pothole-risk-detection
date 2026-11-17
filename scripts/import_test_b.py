#!/usr/bin/env python
"""Validate an exported YOLO annotation folder (class 0 pothole, class 1 MTW) and place it as Test B.

  python scripts/import_test_b.py --annotations exports/cvat_yolo --provenance data/videos/PROVENANCE.csv
Frames are named '<clip_id>_<frame>.jpg'; all frames of a clip go to Test B together (a clip is never split).
Reports the number of DISTINCT source videos (needs >= test_b.min_videos; the real number is printed).
--check-leak also compares against data/processed/pothole/{train,val} by perceptual hash.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import import_test_b, leak_check
from ptr.logging_setup import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="data/processed/test_b")
    ap.add_argument("--provenance", default="data/videos/PROVENANCE.csv")
    ap.add_argument("--check-leak", default=None, help="processed pothole root, e.g. data/processed/pothole")
    ap.add_argument("--strict", action="store_true", help="exit 2 if fewer than min_videos distinct videos")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    setup_logging()
    dp = load_yaml(a.data_prep_config)
    tb = dp["test_b"]
    rep = import_test_b(Path(a.annotations), Path(a.out), Path(a.provenance), dp["clean"]["image_extensions"],
                        tb["allowed_class_ids"], tb["min_box_px"], tb["min_videos"])
    printable = {k: v for k, v in rep.items() if k != "problems"}
    print(json.dumps(printable, indent=2))
    if not rep["ok"]:
        print("VALIDATION FAILED:\n" + "\n".join(rep["problems"][:30]), file=sys.stderr)
        return 3
    if not rep["enough_videos"]:
        print(f"WARNING: only {rep['n_distinct_videos']} distinct videos (need >= {rep['min_videos']}).", file=sys.stderr)
        if a.strict:
            return 2
    if not rep["videos_from_provenance"]:
        print("NOTE: video count falls back to clip count because PROVENANCE.csv has no matching urls.", file=sys.stderr)
    if a.check_leak:
        root = Path(a.check_leak)
        hits = leak_check(Path(a.out) / "images", [root / "train" / "images", root / "val" / "images"],
                          dp["dedupe"]["hash_size"], dp["dedupe"]["hamming_threshold"], dp["clean"]["image_extensions"])
        print(f"leak check: {len(hits)} near-duplicate pairs between Test B and train/val")
        for h in hits[:20]:
            print("  ", h)
        if hits:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
