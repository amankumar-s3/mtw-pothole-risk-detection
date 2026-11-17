#!/usr/bin/env python
"""Like make_splits.py, but holds out only some WHOLE VIDEOS of the test source (the rest joins train/val)."""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ptr.config import load_yaml
from ptr.dataprep import split_stats, write_dataset_yaml, write_splits
from ptr.logging_setup import setup_logging
from ptr.utils import seed_everything, write_csv


def video_key(filename, source):
    stem = Path(filename).stem
    if stem.startswith(f"{source}__"):
        stem = stem[len(source) + 2:]
    return re.sub(r"\d{8}$", "", stem)


def pick_test_videos(clusters, source, fraction, seed):
    per = Counter(video_key(r["file"], source) for r in clusters if r["source"] == source)
    keys = sorted(per)
    random.Random(seed).shuffle(keys)
    total, got, chosen = sum(per.values()), 0, []
    for k in keys:
        if got >= fraction * total:
            break
        chosen.append(k)
        got += per[k]
    return sorted(chosen)


def assign(clusters, source, test_videos, val_fraction, seed):
    rng = random.Random(seed)
    key = lambda r: f"{r['source']}/{r['file']}"
    tv = set(test_videos)
    in_test = lambda r: r["source"] == source and video_key(r["file"], source) in tv
    test_clusters = {r["cluster_id"] for r in clusters if in_test(r)}
    split, removed, pool = {}, 0, []
    for r in clusters:
        if in_test(r):
            split[key(r)] = "test_a"
        elif r["cluster_id"] in test_clusters:
            removed += 1
        else:
            pool.append(r)
    groups = defaultdict(list)
    for r in pool:
        groups[r["cluster_id"]].append(r)
    totals = Counter(r["source"] for r in pool)
    target = {s: val_fraction * n for s, n in totals.items()}
    val_count = Counter()
    gids = sorted(groups)
    rng.shuffle(gids)
    for g in gids:
        members = groups[g]
        major = Counter(m["source"] for m in members).most_common(1)[0][0]
        to_val = val_count[major] < target[major]
        for m in members:
            split[key(m)] = "val" if to_val else "train"
        if to_val:
            val_count[major] += len(members)
    return split, removed


def check(manifest, source, test_videos):
    problems, tv = [], set(test_videos)
    by_cluster = defaultdict(set)
    for r in manifest:
        by_cluster[str(r["cluster_id"])].add(r["split"])
        if r["source"] == source:
            if (video_key(r["file"], source) in tv) != (r["split"] == "test_a"):
                problems.append(f"video split between test_a and train/val: {r['file']}")
        elif r["split"] == "test_a":
            problems.append(f"non-test source in test_a: {r['file']}")
    problems += [f"cluster {c} spans splits {sorted(s)}" for c, s in by_cluster.items() if len(s) > 1]
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--processed", default="data/processed/pothole")
    ap.add_argument("--dataset-yaml", default="configs/dataset_pothole.yaml")
    ap.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    ap.add_argument("--test-a-source", default="bucko")
    ap.add_argument("--test-video-fraction", type=float, default=0.4)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_yaml(a.data_prep_config)
    vf = a.val_fraction if a.val_fraction is not None else cfg["splits"]["val_fraction"]
    seed = a.seed if a.seed is not None else cfg["splits"]["seed"]
    exts = cfg["clean"]["image_extensions"]
    seed_everything(seed)
    interim, processed = Path(a.interim), Path(a.processed)
    cl_path = interim / "clusters.csv"
    if not cl_path.is_file():
        print("run scripts/clean_dedupe.py first (clusters.csv missing)", file=sys.stderr)
        return 2
    with open(cl_path, newline="", encoding="utf-8") as fh:
        clusters = [{**r, "cluster_id": int(r["cluster_id"])} for r in csv.DictReader(fh)]
    if not any(r["source"] == a.test_a_source for r in clusters):
        print(f"source '{a.test_a_source}' not found", file=sys.stderr)
        return 2
    tvids = pick_test_videos(clusters, a.test_a_source, a.test_video_fraction, seed)
    split, removed = assign(clusters, a.test_a_source, tvids, vf, seed)
    manifest = write_splits(interim, processed, clusters, split, exts)
    problems = check(manifest, a.test_a_source, tvids)
    if problems:
        print("LEAKAGE CHECK FAILED:\n" + "\n".join(problems[:20]), file=sys.stderr)
        return 3
    write_csv(processed / "manifest.csv", manifest, ["file", "source", "split", "cluster_id"])
    write_dataset_yaml(Path(a.dataset_yaml), processed)
    st = split_stats(processed, manifest, exts)
    lines = ["# Split statistics (video-level hold-out)", "",
             f"- Test A = whole videos of `{a.test_a_source}` only: {', '.join(tvids)}",
             f"- the other `{a.test_a_source}` videos are in train/val",
             "- caution: held-out videos may show the same streets as training videos, so Test A is easier "
             "than a truly unseen source",
             f"- val fraction target: {vf}, seed {seed}",
             f"- near-duplicates of Test A images removed from train/val: {removed}", "",
             "| source | split | images | boxes |", "|---|---|---|---|"]
    for s, d in sorted(st["counts"].items()):
        for k, v in sorted(d.items()):
            lines.append(f"| {s} | {k} | {v['images']} | {v['boxes']} |")
    (processed / "stats.md").write_text("\n".join(lines) + "\n")
    print(f"test videos: {tvids}\nsplits written to {processed}; no video or hash cluster spans splits (checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
