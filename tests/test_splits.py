import csv
from collections import Counter
from pathlib import Path

import pytest

import clean_dedupe
import convert_to_yolo
import make_splits
from dataset_helpers import make_img, make_voc_source, make_yolo_source, write_voc, write_yolo
from ptr.dataprep import assign_splits, check_no_leak, read_yolo_txt

ROOT = Path(__file__).resolve().parents[1]
DP = str(ROOT / "configs" / "data_prep.yaml")


def build(tmp_path):
    raw = tmp_path / "raw"
    make_voc_source(raw / "rdd_a", "a", 12, 100)
    make_voc_source(raw / "rdd_b", "b", 12, 200)
    make_yolo_source(raw / "bucko", "k", 8, 300)
    # near-duplicates inside a train source (same hash cluster must stay together)
    for i in range(3):
        make_img(raw / "rdd_a" / "train" / "images" / f"dupa_{i}.jpg", 100, noise=2.0 + i)
        write_voc(raw / "rdd_a" / "train" / "annotations" / "xmls" / f"dupa_{i}.xml", 96, 72, [("D40", 10, 10, 50, 40)])
    # a near-duplicate of a Test A image hiding in another source (leak!)
    make_img(raw / "rdd_b" / "train" / "images" / "leak.jpg", 300, noise=2.0)
    write_voc(raw / "rdd_b" / "train" / "annotations" / "xmls" / "leak.xml", 96, 72, [("D40", 10, 10, 50, 40)])
    interim = tmp_path / "interim"
    for name, fmt_map in (("rdd_a", "rdd2022"), ("rdd_b", "rdd2022"), ("bucko", "bucko")):
        assert convert_to_yolo.main(["--source", name, "--input", str(raw / name),
                                     "--class-map", str(ROOT / "configs" / "class_maps" / f"{fmt_map}.yaml"),
                                     "--out-root", str(interim), "--data-prep-config", DP]) == 0
    assert clean_dedupe.main(["--interim", str(interim), "--data-prep-config", DP]) == 0
    processed = tmp_path / "processed" / "pothole"
    ds_yaml = tmp_path / "dataset.yaml"
    rc = make_splits.main(["--interim", str(interim), "--processed", str(processed), "--dataset-yaml", str(ds_yaml),
                           "--data-prep-config", DP, "--test-a-source", "bucko", "--seed", "0"])
    assert rc == 0
    with open(processed / "manifest.csv", newline="") as fh:
        manifest = list(csv.DictReader(fh))
    return processed, manifest, ds_yaml


def test_no_hash_cluster_spans_splits_and_test_a_is_held_out(tmp_path):
    processed, manifest, ds_yaml = build(tmp_path)
    assert check_no_leak(manifest, "bucko") == []
    by_source = {s: {r["split"] for r in manifest if r["source"] == s} for s in {r["source"] for r in manifest}}
    assert by_source["bucko"] == {"test_a"}
    assert "test_a" not in by_source["rdd_a"] | by_source["rdd_b"]
    # the planted duplicate of a Test A image was removed from train/val
    assert "rdd_b__leak.jpg" not in {r["file"] for r in manifest}
    # near-duplicates of the same image stay together
    dup_splits = {r["split"] for r in manifest if r["file"].startswith("rdd_a__dupa_") or r["file"] == "rdd_a__a_000.jpg"}
    assert len(dup_splits) == 1


def test_every_label_is_class_zero_and_counts_match_files(tmp_path):
    processed, manifest, _ = build(tmp_path)
    for sp in ("train", "val", "test_a"):
        imgs = {p.stem for p in (processed / sp / "images").glob("*.jpg")}
        labs = {p.stem for p in (processed / sp / "labels").glob("*.txt")}
        assert imgs == labs
        assert imgs == {Path(r["file"]).stem for r in manifest if r["split"] == sp}
        for lp in (processed / sp / "labels").glob("*.txt"):
            assert {r[0] for r in read_yolo_txt(lp)} <= {0}
    counts = Counter(r["split"] for r in manifest)
    assert counts["test_a"] == 8 and counts["train"] > counts["val"] > 0


def test_val_fraction_is_roughly_respected_per_source(tmp_path):
    _, manifest, _ = build(tmp_path)
    for s in ("rdd_a", "rdd_b"):
        rows = [r for r in manifest if r["source"] == s]
        frac = sum(r["split"] == "val" for r in rows) / len(rows)
        assert 0.05 <= frac <= 0.35


def test_dataset_yaml_and_stats_written(tmp_path):
    processed, _, ds = build(tmp_path)
    text = ds.read_text()
    assert "pothole" in text and "test_a/images" in text
    assert (processed / "stats.md").exists() and (processed / "manifest.csv").exists()


def test_splits_are_deterministic_for_a_seed(tmp_path):
    clusters = [{"file": f"f{i}.jpg", "source": "s", "cluster_id": i // 2} for i in range(40)]
    a, _ = assign_splits(clusters, "none", 0.25, 7)
    b, _ = assign_splits(clusters, "none", 0.25, 7)
    c, _ = assign_splits(clusters, "none", 0.25, 8)
    assert a == b and a != c


def test_leak_checker_catches_violations():
    bad = [{"file": "a", "source": "x", "split": "train", "cluster_id": 1},
           {"file": "b", "source": "x", "split": "val", "cluster_id": 1},
           {"file": "c", "source": "bucko", "split": "train", "cluster_id": 2},
           {"file": "d", "source": "x", "split": "test_a", "cluster_id": 3}]
    probs = check_no_leak(bad, "bucko")
    assert any("spans splits" in p for p in probs)
    assert any("outside test_a" in p for p in probs) and any("non-test_a source" in p for p in probs)


def test_make_splits_fails_without_clusters_or_test_source(tmp_path):
    assert make_splits.main(["--interim", str(tmp_path), "--processed", str(tmp_path / "p"),
                             "--dataset-yaml", str(tmp_path / "d.yaml"), "--data-prep-config", DP]) == 2
