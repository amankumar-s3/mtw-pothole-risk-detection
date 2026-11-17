#!/usr/bin/env python
"""Audit a downloaded dataset: counts, class histogram, box-size percentiles. Writes docs/audits/<name>.md.

  python scripts/verify_datasets.py --root data/raw/rdd2022_india --format voc --name rdd2022_india
Measures only; it never fills in licences or URLs (those stay TO VERIFY in docs/data_sources.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import audit_dataset, audit_markdown
from ptr.logging_setup import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--format", choices=["voc", "yolo"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="docs/audits")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    setup_logging()
    exts = load_yaml(a.data_prep_config)["clean"]["image_extensions"]
    if not Path(a.root).is_dir():
        print(f"root not found: {a.root}", file=sys.stderr)
        return 2
    res = audit_dataset(Path(a.root), a.format, exts)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.md").write_text(audit_markdown(a.name, res))
    (out / f"{a.name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "corrupt_images"}, indent=2))
    print(f"wrote {out / (a.name + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
