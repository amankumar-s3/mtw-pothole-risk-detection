#!/usr/bin/env python
"""Leave-source-out splits grouped by perceptual-hash cluster (never random frame-level).

  python scripts/make_splits.py --test-a-source bucko
Test A = the WHOLE held-out source (default from configs/data_prep.yaml). The remaining sources are split
train/val (~85/15 per source) so that no hash cluster crosses splits. Near-duplicates of Test A images are
removed from train/val. Writes data/processed/pothole/{train,val,test_a}/{images,labels}, manifest.csv,
stats.md, box_size_hist.png and configs/dataset_pothole.yaml. Test B (data/processed/test_b) is untouched.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import (assign_splits, check_no_leak, split_stats, write_dataset_yaml, write_splits)
from ptr.logging_setup import setup_logging
from ptr.utils import seed_everything, write_csv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--processed", default="data/processed/pothole")
    ap.add_argument("--dataset-yaml", default="configs/dataset_pothole.yaml")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    ap.add_argument("--test-a-source", default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    log = setup_logging()
    sp = load_yaml(a.data_prep_config)["splits"]
    exts = load_yaml(a.data_prep_config)["clean"]["image_extensions"]
    test_a = a.test_a_source or sp["test_a_source"]
    vf = a.val_fraction if a.val_fraction is not None else sp["val_fraction"]
    seed = a.seed if a.seed is not None else sp["seed"]
    seed_everything(seed)
    interim, processed = Path(a.interim), Path(a.processed)
    cl_path = interim / "clusters.csv"
    if not cl_path.is_file():
        print("run scripts/clean_dedupe.py first (clusters.csv missing)", file=sys.stderr)
        return 2
    with open(cl_path, newline="", encoding="utf-8") as fh:
        clusters = [{**r, "cluster_id": int(r["cluster_id"])} for r in csv.DictReader(fh)]
    if not any(r["source"] == test_a for r in clusters):
        print(f"held-out source '{test_a}' not found in {interim}; sources: {sorted({r['source'] for r in clusters})}",
              file=sys.stderr)
        return 2
    split, info = assign_splits(clusters, test_a, vf, seed)
    manifest = write_splits(interim, processed, clusters, split, exts)
    problems = check_no_leak(manifest, test_a)
    if problems:
        print("LEAKAGE CHECK FAILED:\n" + "\n".join(problems[:20]), file=sys.stderr)
        return 3
    write_csv(processed / "manifest.csv", manifest, ["file", "source", "split", "cluster_id"])
    write_dataset_yaml(Path(a.dataset_yaml), processed)
    st = split_stats(processed, manifest, exts)
    lines = ["# Split statistics", "", f"- held-out Test A source: `{test_a}`", f"- val fraction target: {vf}, seed {seed}",
             f"- near-duplicates of Test A images removed from train/val: {info['leak_removed']}", "",
             "| source | split | images | boxes |", "|---|---|---|---|"]
    for s, d in sorted(st["counts"].items()):
        for k, v in sorted(d.items()):
            lines.append(f"| {s} | {k} | {v['images']} | {v['boxes']} |")
    (processed / "stats.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for k, v in sorted(st["sizes"].items()):
            if v:
                ax.hist(v, bins=40, alpha=0.5, label=f"{k} (n={len(v)})", density=True)
        ax.set_xlabel("sqrt(relative box area)")
        ax.set_ylabel("density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(processed / "box_size_hist.png", dpi=120)
    except Exception as exc:                         # plotting must never break the split
        log.warning("histogram not written: %s", exc)
    print(f"splits written to {processed}; no hash cluster spans splits (checked). See {processed}/stats.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
