import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import check_agreement
import extract_frames
import import_test_b
import preview_annotations
import prelabel
import verify_datasets
from dataset_helpers import make_img, make_voc_source, make_yolo_source, write_voc, write_yolo
from ptr.dataprep import (annotator_agreement, box_iou_xywhn, import_test_b as do_import, leak_check,
                          split_clip_id, write_prelabels, read_yolo_txt)

ROOT = Path(__file__).resolve().parents[1]
DP = str(ROOT / "configs" / "data_prep.yaml")
EXTS = [".jpg"]


def test_verify_datasets_voc(tmp_path):
    make_voc_source(tmp_path / "d", "a", 5, 1)
    make_img(tmp_path / "d" / "train" / "images" / "noann.jpg", 50)                # image without annotation
    write_voc(tmp_path / "d" / "train" / "annotations" / "xmls" / "orphan.xml", 96, 72, [("D40", 1, 1, 9, 9)])
    (tmp_path / "d" / "train" / "images" / "bad.jpg").write_bytes(b"xx")
    assert verify_datasets.main(["--root", str(tmp_path / "d"), "--format", "voc", "--name", "t",
                                 "--out-dir", str(tmp_path / "audits"), "--data-prep-config", DP]) == 0
    r = json.loads((tmp_path / "audits" / "t.json").read_text())
    assert r["n_boxes"] == 5 and r["class_histogram"] == {"D40": 5}
    assert r["images_without_annotation"] == 2 and r["annotations_without_image"] == 1   # noann.jpg + bad.jpg
    assert len(r["corrupt_images"]) == 1
    assert abs(r["rel_width_percentiles"]["p50"] - 40 / 96) < 1e-3
    assert "Class histogram" in (tmp_path / "audits" / "t.md").read_text()


def test_verify_datasets_yolo_and_missing_root(tmp_path):
    make_yolo_source(tmp_path / "y", "k", 4, 9)
    assert verify_datasets.main(["--root", str(tmp_path / "y"), "--format", "yolo", "--name", "y",
                                 "--out-dir", str(tmp_path / "a"), "--data-prep-config", DP]) == 0
    assert json.loads((tmp_path / "a" / "y.json").read_text())["class_histogram"] == {"0": 4}
    assert verify_datasets.main(["--root", str(tmp_path / "none"), "--format", "yolo", "--name", "n",
                                 "--out-dir", str(tmp_path / "a"), "--data-prep-config", DP]) == 2


def test_preview_contact_sheet(tmp_path):
    make_voc_source(tmp_path / "d", "a", 8, 1)
    out = tmp_path / "sheet.png"
    assert preview_annotations.main(["--root", str(tmp_path / "d"), "--format", "voc", "--n", "6", "--seed", "1",
                                     "--out", str(out), "--data-prep-config", DP]) == 0
    img = cv2.imread(str(out))
    assert img is not None and img.shape[0] > 0 and img.shape[1] == 5 * 256
    out2 = tmp_path / "sheet2.png"
    preview_annotations.main(["--root", str(tmp_path / "d"), "--format", "voc", "--n", "6", "--seed", "1",
                              "--out", str(out2), "--data-prep-config", DP])
    assert out.read_bytes() == out2.read_bytes()                                    # seeded


def test_extract_frames_skips_blurry(tmp_path):
    vid = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    rng = np.random.default_rng(0)
    for i in range(60):
        if 20 <= i < 30:
            f = np.full((48, 64, 3), 128, np.uint8)                                # flat = "blurry"
        else:
            f = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)                  # sharp noise
        w.write(f)
    w.release()
    out = tmp_path / "cands"
    assert extract_frames.main(["--video", str(vid), "--clip-id", "clipX", "--start", "0", "--end", "5.9",
                                "--interval", "1.0", "--out-root", str(out), "--blur-threshold", "50",
                                "--data-prep-config", DP]) == 0
    files = sorted((out / "clipX").glob("*.jpg"))
    names = [f.name for f in files]
    assert all(n.startswith("clipX_") for n in names)
    assert len(files) == 5                                    # samples at frames 0,10,...,50; only frame 20 is flat
    assert "clipX_000020.jpg" not in names and "clipX_000030.jpg" in names


def test_prelabel_writes_editable_yolo(tmp_path):
    make_img(tmp_path / "imgs" / "c_000001.jpg", 1)
    calls = []

    def fake(path):
        calls.append(path.name)
        return [(0, 0.9, (10.0, 10.0, 50.0, 40.0)), (1, 0.8, (60.0, 20.0, 90.0, 70.0)), (0, 0.7, (500, 500, 600, 600))]
    rep = write_prelabels(tmp_path / "imgs", tmp_path / "out", fake, EXTS)
    rows = read_yolo_txt(tmp_path / "out" / "c_000001.txt")
    assert rep == {"images": 1, "boxes": 2} and [r[0] for r in rows] == [0, 1]
    assert prelabel.main(["--images", str(tmp_path), "--out", str(tmp_path / "o")]) == 2        # needs weights


def make_export(root: Path, clips: dict[str, int], bad: bool = False):
    seed = 0
    for cid, n in clips.items():
        for f in range(n):
            seed += 1
            stem = f"{cid}_{f:06d}"
            make_img(root / f"{stem}.jpg", seed)
            write_yolo(root / f"{stem}.txt", [(0, 0.5, 0.6, 0.2, 0.2), (1, 0.3, 0.4, 0.15, 0.3)])
    if bad:
        write_yolo(root / "bad_000001.txt", [(5, 0.5, 0.5, 0.2, 0.2)])
        make_img(root / "bad_000001.jpg", 999)


def provenance(path: Path, mapping: dict[str, str]):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "source", "url"])
        for c, u in mapping.items():
            w.writerow([c, "youtube", u])


def test_split_clip_id():
    assert split_clip_id("indiaRoad_clip_02_000123") == "indiaRoad_clip_02"
    assert split_clip_id("plain") == "plain"


def test_import_test_b_counts_distinct_videos_and_keeps_clips_together(tmp_path):
    exp = tmp_path / "exp"
    make_export(exp, {f"c{i}": 2 for i in range(1, 11)})
    prov = tmp_path / "prov.csv"
    provenance(prov, {f"c{i}": f"https://example.invalid/v{(i + 1) // 2}" for i in range(1, 11)})   # 10 clips, 5 videos
    out = tmp_path / "tb"
    rep = do_import(exp, out, prov, EXTS, [0, 1], 4, 8)
    assert rep["ok"] and rep["n_images"] == 20 and rep["n_clips"] == 10
    assert rep["n_distinct_videos"] == 5 and rep["enough_videos"] is False        # honest count, not the clip count
    assert len(list((out / "images").glob("*.jpg"))) == len(list((out / "labels").glob("*.txt"))) == 20
    rows = list(csv.DictReader(open(out / "clips.csv")))
    assert {r["clip_id"] for r in rows} == {f"c{i}" for i in range(1, 11)}
    assert all(int(r["n_images"]) == 2 for r in rows)


def test_import_test_b_rejects_bad_class_and_script_exit_codes(tmp_path):
    exp = tmp_path / "exp"
    make_export(exp, {"c1": 1}, bad=True)
    rc = import_test_b.main(["--annotations", str(exp), "--out", str(tmp_path / "tb"), "--provenance",
                             str(tmp_path / "nope.csv"), "--data-prep-config", DP])
    assert rc == 3 and not (tmp_path / "tb" / "images").exists()
    exp2 = tmp_path / "exp2"
    make_export(exp2, {"c1": 1})
    assert import_test_b.main(["--annotations", str(exp2), "--out", str(tmp_path / "tb2"), "--provenance",
                               str(tmp_path / "nope.csv"), "--strict", "--data-prep-config", DP]) == 2


def test_import_test_b_rejects_out_of_range_boxes(tmp_path):
    exp = tmp_path / "exp"
    make_img(exp / "c1_000001.jpg", 3)
    write_yolo(exp / "c1_000001.txt", [(0, 0.95, 0.5, 0.3, 0.3)])                  # sticks out of the image
    rep = do_import(exp, tmp_path / "tb", None, EXTS, [0, 1], 4, 8)
    assert not rep["ok"] and any("outside" in p for p in rep["problems"])


def test_leak_check_between_test_b_and_train(tmp_path):
    make_img(tmp_path / "tb" / "t1.jpg", 5)
    make_img(tmp_path / "tb" / "t2.jpg", 6)
    make_img(tmp_path / "tr" / "x1.jpg", 5, noise=2.0)                            # near-duplicate of t1
    make_img(tmp_path / "tr" / "x2.jpg", 77)
    hits = leak_check(tmp_path / "tb", [tmp_path / "tr"], 8, 4, EXTS)
    assert hits == [("t1.jpg", "x1.jpg")]


def test_box_iou_and_agreement_hand_computed(tmp_path):
    assert box_iou_xywhn((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) == pytest.approx(1.0)
    assert box_iou_xywhn((0.2, 0.2, 0.2, 0.2), (0.8, 0.8, 0.2, 0.2)) == 0.0
    # half-overlapping boxes: intersection 0.1*0.2 = 0.02, union 0.04 + 0.04 - 0.02 = 0.06 -> 1/3
    assert box_iou_xywhn((0.5, 0.5, 0.2, 0.2), (0.6, 0.5, 0.2, 0.2)) == pytest.approx(1 / 3)
    A, B = tmp_path / "A", tmp_path / "B"
    write_yolo(A / "f1.txt", [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.2, 0.2, 0.1, 0.1)])
    write_yolo(B / "f1.txt", [(0, 0.51, 0.5, 0.2, 0.2), (0, 0.9, 0.9, 0.1, 0.1)])     # pothole match + 1 extra; MTW missing
    write_yolo(A / "f2.txt", [(0, 0.3, 0.3, 0.2, 0.2)])
    write_yolo(B / "f2.txt", [(0, 0.7, 0.7, 0.2, 0.2)])                                # no overlap
    write_yolo(A / "only_a.txt", [])
    r = annotator_agreement(A, B, 0.5)
    assert r["n_common_files"] == 2 and r["only_in_a"] == ["only_a"]
    p = r["per_class"][0]
    assert (p["a"], p["b"], p["matched"]) == (2, 3, 1) and p["agreement_f1"] == pytest.approx(2 * 1 / 5)
    m = r["per_class"][1]
    assert (m["a"], m["b"], m["matched"]) == (1, 0, 0) and m["agreement_f1"] == 0.0


def test_check_agreement_script_writes_report(tmp_path):
    A, B = tmp_path / "A", tmp_path / "B"
    write_yolo(A / "f.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
    write_yolo(B / "f.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
    assert check_agreement.main(["--a", str(A), "--b", str(B), "--out-dir", str(tmp_path / "o"), "--name", "t",
                                 "--data-prep-config", DP]) == 0
    assert "1.000" in (tmp_path / "o" / "agreement_t.md").read_text()
