#!/usr/bin/env python
"""Annotator agreement between two YOLO label folders (boxes matched at IoU 0.5), per class.

  python scripts/check_agreement.py --a labels_annotator1 --b labels_annotator2
Use it on the 20 double-annotated frames (docs/annotation_guide.md). Writes eval_results/agreement_<name>.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import annotator_agreement
from ptr.logging_setup import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--name", default="annotators")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    a = ap.parse_args(argv)
    setup_logging()
    thr = a.iou if a.iou is not None else load_yaml(a.data_prep_config)["agreement"]["iou_threshold"]
    r = annotator_agreement(Path(a.a), Path(a.b), thr)
    lines = [f"# Annotator agreement: {a.name}", "", f"- common label files: {r['n_common_files']}",
             f"- only in A: {len(r['only_in_a'])}, only in B: {len(r['only_in_b'])}", f"- IoU threshold: {thr}", "",
             "| class | boxes A | boxes B | matched | agreement F1 |", "|---|---|---|---|---|"]
    for c, s in r["per_class"].items():
        lines.append(f"| {c} | {s['a']} | {s['b']} | {s['matched']} | {s['agreement_f1']:.3f} |")
    lines += ["", "Small samples give noisy agreement numbers: always quote the counts above."]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"agreement_{a.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
