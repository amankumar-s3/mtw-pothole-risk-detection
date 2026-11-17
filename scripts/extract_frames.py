#!/usr/bin/env python
"""Sample frames from a public video for Test B annotation (about 1 per second), skipping blurry frames.

  python scripts/extract_frames.py --video data/videos/clip01.mp4 --clip-id clip01 --start 5 --end 60
Output: data/interim/test_b_candidates/<clip_id>/<clip_id>_<frame>.jpg  (the video itself is never committed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import extract_frames
from ptr.logging_setup import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--interval", type=float, default=None, help="seconds between frames (default: config)")
    ap.add_argument("--blur-threshold", type=float, default=None)
    ap.add_argument("--out-root", default="data/interim/test_b_candidates")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    setup_logging()
    fr = load_yaml(a.data_prep_config)["frames"]
    rep = extract_frames(Path(a.video), a.clip_id, a.start, a.end, a.interval or fr["interval_seconds"],
                         Path(a.out_root) / a.clip_id, a.blur_threshold if a.blur_threshold is not None else fr["blur_threshold"],
                         fr["jpeg_quality"])
    print(f"{a.clip_id}: kept {rep['kept']}, skipped blurry {rep['skipped_blurry']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
