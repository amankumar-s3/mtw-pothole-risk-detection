#!/usr/bin/env python
"""Convert one source (Pascal VOC XML or YOLO txt) to the unified YOLO layout using a class-map YAML.

  python scripts/convert_to_yolo.py --source rdd2022_india --input data/raw/rdd2022_india \\
      --class-map configs/class_maps/rdd2022.yaml
Output: data/interim/<source>/{images,labels}, files prefixed '<source>__'. Pothole = class 0, all else dropped.
Re-running rebuilds that source's folder (idempotent).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import convert_source
from ptr.logging_setup import setup_logging
from ptr.utils import seed_everything


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="name used as filename prefix and interim folder (e.g. bucko)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--out-root", default="data/interim")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rel-width", type=float, default=None)
    ap.add_argument("--max-rel-area", type=float, default=None)
    a = ap.parse_args(argv)
    setup_logging()
    seed_everything(a.seed)
    cm = load_yaml(a.class_map)
    dp = load_yaml(a.data_prep_config)
    fmt = cm["format"]
    mapping = cm["class_map"] if fmt == "voc" else cm["class_id_map"]
    if not Path(a.input).is_dir():
        print(f"input not found: {a.input}", file=sys.stderr)
        return 2
    rep = convert_source(Path(a.input), Path(a.out_root) / a.source, a.source, fmt, mapping,
                         dp["clean"]["image_extensions"], dp["convert"]["keep_negative_ratio"],
                         dp["convert"]["exif_policy"], a.seed, cm.get("case_insensitive", True),
                         a.max_rel_width, a.max_rel_area)
    print(json.dumps(rep, indent=2))
    if rep.get("images_written", 0) == 0:
        print("WARNING: nothing was written. Check the class map against the histogram from verify_datasets.py.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
