#!/usr/bin/env python
"""Remove corrupt images and tiny boxes, then cluster near-duplicates (perceptual hash) across ALL sources.

  python scripts/clean_dedupe.py --interim data/interim
Writes data/interim/clusters.csv (file, source, cluster_id, phash) and data/interim/cleaning_report.csv.
Clusters are used by make_splits.py so that near-duplicates never end up in different splits.
Use --drop-duplicates to keep only one image per cluster within the same source (off by default).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import clean_source, compute_clusters
from ptr.logging_setup import setup_logging
from ptr.utils import write_csv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    ap.add_argument("--drop-duplicates", action="store_true")
    a = ap.parse_args(argv)
    log = setup_logging()
    dp = load_yaml(a.data_prep_config)
    exts = dp["clean"]["image_extensions"]
    root = Path(a.interim)
    sources = sorted(p for p in root.iterdir() if (p / "images").is_dir()) if root.is_dir() else []
    if not sources:
        print(f"no converted sources under {root}", file=sys.stderr)
        return 2
    report = []
    for sd in sources:
        report += clean_source(sd, dp["clean"]["min_box_px"], dp["clean"]["tiny_box_policy"], exts)
    clusters = compute_clusters(root, dp["dedupe"]["hash_size"], dp["dedupe"]["hamming_threshold"], exts)
    if a.drop_duplicates:
        seen: set[tuple[str, int]] = set()
        keep = []
        for r in clusters:
            k = (r["source"], r["cluster_id"])
            if k in seen:
                (root / r["source"] / "images" / r["file"]).unlink(missing_ok=True)
                (root / r["source"] / "labels" / (Path(r["file"]).stem + ".txt")).unlink(missing_ok=True)
                report.append({"file": r["file"], "source": r["source"], "action": "removed_duplicate", "detail": f"cluster {r['cluster_id']}"})
            else:
                seen.add(k)
                keep.append(r)
        clusters = keep
    write_csv(root / "clusters.csv", clusters, ["file", "source", "cluster_id", "phash"])
    write_csv(root / "cleaning_report.csv", report, ["file", "source", "action", "detail"])
    n_clusters = len({r["cluster_id"] for r in clusters})
    log.info("%d images, %d clusters, %d cleaning actions", len(clusters), n_clusters, len(report))
    print(f"{len(clusters)} images in {n_clusters} clusters; {len(report)} cleaning actions -> {root}/cleaning_report.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
