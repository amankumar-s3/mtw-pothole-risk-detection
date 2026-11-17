import json

from dataset_helpers import make_img, write_voc, write_yolo
from ptr.dataprep import (clean_source, convert_source, hamming_clusters, parse_voc, phash_int, read_yolo_txt,
                          xyxy_to_yolo)

EXTS = [".jpg", ".jpeg", ".png"]


def labels(out):
    return {p.name: read_yolo_txt(p) for p in (out / "labels").glob("*.txt")}


def test_voc_conversion_maps_drops_clamps_and_prefixes(tmp_path):
    src = tmp_path / "rdd"
    make_img(src / "train" / "images" / "a.jpg", 1)
    write_voc(src / "train" / "annotations" / "a.xml", 96, 72, [
        ("D40", 10, 10, 50, 40),            # pothole
        (" d40 ", 60, 20, 90, 60),          # casing / whitespace
        ("D00", 5, 5, 30, 30),              # crack: dropped
        ("D40", -20, 50, 20, 100),          # partly outside: clamped
        ("D40", 200, 200, 300, 300),        # fully outside: invalid
    ])
    make_img(src / "train" / "images" / "b.jpg", 2)                 # image without xml (test-like): skipped
    make_img(src / "train" / "images" / "neg.jpg", 3)
    write_voc(src / "train" / "annotations" / "neg.xml", 96, 72, [("D00", 1, 1, 20, 20)])   # only a crack
    out = tmp_path / "interim" / "rdd"
    rep = convert_source(src, out, "rdd", "voc", {"D40": 0}, EXTS, 0.0, "keep", 0)
    assert rep["images_written"] == 1 and rep["unlabeled_images_skipped"] == 1
    assert rep["boxes_dropped_class"] == 2 and rep["boxes_dropped_invalid"] == 1 and rep["boxes_clamped"] == 1
    assert rep["empty_images_dropped"] == 1 and rep["dropped_class_histogram"] == {"D00": 2}
    lab = labels(out)
    assert list(lab) == ["rdd__a.txt"] and (out / "images" / "rdd__a.jpg").exists()
    assert len(lab["rdd__a.txt"]) == 3 and {r[0] for r in lab["rdd__a.txt"]} == {0}
    assert all(0 <= v <= 1 for r in lab["rdd__a.txt"] for v in r[1:])
    assert (out / "convert_report.json").exists()


def test_negatives_kept_on_request(tmp_path):
    src = tmp_path / "s"
    make_img(src / "x.jpg", 1)
    write_voc(src / "x.xml", 96, 72, [("D00", 1, 1, 20, 20)])
    out = tmp_path / "o"
    rep = convert_source(src, out, "s", "voc", {"D40": 0}, EXTS, 1.0, "keep", 0)
    assert rep["negative_images_kept"] == 1 and labels(out)["s__x.txt"] == []


def test_xml_size_is_used_for_normalisation(tmp_path):
    src = tmp_path / "s"
    make_img(src / "x.jpg", 1, size=(192, 144))                    # image twice as large as the XML says
    write_voc(src / "x.xml", 96, 72, [("D40", 0, 0, 48, 36)])
    out = tmp_path / "o"
    rep = convert_source(src, out, "s", "voc", {"D40": 0}, EXTS, 0.0, "keep", 0)
    assert rep["xml_size_differs_from_image"] == 1
    c, xc, yc, w, h = labels(out)["s__x.txt"][0]
    assert (round(xc, 3), round(yc, 3), round(w, 3), round(h, 3)) == (0.25, 0.25, 0.5, 0.5)


def test_yolo_conversion_with_id_map_and_crlf(tmp_path):
    src = tmp_path / "bucko"
    make_img(src / "images" / "f1.jpg", 4)
    write_yolo(src / "images" / "f1.txt", [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.2, 0.2, 0.1, 0.1), (0, 0.99, 0.5, 0.2, 0.2)], crlf=True)
    (src / "images" / "classes.txt").write_text("pothole\n")           # must not be treated as a label
    out = tmp_path / "o"
    rep = convert_source(src, out, "bucko", "yolo", {0: 0}, EXTS, 0.0, "keep", 0)
    rows = labels(out)["bucko__f1.txt"]
    assert rep["boxes_dropped_class"] == 1 and len(rows) == 2 and {r[0] for r in rows} == {0}
    assert rows[1][1] + rows[1][3] / 2 <= 1.0 + 1e-6                     # clamped to the image


def test_conversion_is_idempotent(tmp_path):
    src = tmp_path / "s"
    make_img(src / "x.jpg", 1)
    write_voc(src / "x.xml", 96, 72, [("D40", 10, 10, 50, 40)])
    out = tmp_path / "o"
    convert_source(src, out, "s", "voc", {"D40": 0}, EXTS, 0.0, "keep", 0)
    first = sorted(p.name for p in out.rglob("*") if p.is_file())
    convert_source(src, out, "s", "voc", {"D40": 0}, EXTS, 0.0, "keep", 0)
    assert first == sorted(p.name for p in out.rglob("*") if p.is_file())


def test_voc_parse_and_yolo_helpers(tmp_path):
    write_voc(tmp_path / "a.xml", 100, 50, [("D40", 10, 10, 30, 20)])
    v = parse_voc(tmp_path / "a.xml")
    assert (v.width, v.height) == (100, 50) and v.objects == [("D40", 10.0, 10.0, 30.0, 20.0)]
    assert xyxy_to_yolo(10, 10, 30, 20, 100, 50) == (0.2, 0.3, 0.2, 0.2)


def test_clean_removes_corrupt_and_tiny(tmp_path):
    d = tmp_path / "src"
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir()
    make_img(d / "images" / "ok.jpg", 1)
    write_yolo(d / "labels" / "ok.txt", [(0, 0.5, 0.5, 0.4, 0.4), (0, 0.2, 0.2, 0.02, 0.02)])        # 2nd is ~2px
    make_img(d / "images" / "tiny.jpg", 2)
    write_yolo(d / "labels" / "tiny.txt", [(0, 0.5, 0.5, 0.02, 0.02)])
    (d / "images" / "bad.jpg").write_bytes(b"not an image")
    write_yolo(d / "labels" / "bad.txt", [(0, 0.5, 0.5, 0.4, 0.4)])
    rep = clean_source(d, 8, "drop_image_if_empty", EXTS)
    actions = {(r["file"], r["action"]) for r in rep}
    assert ("bad.jpg", "removed_corrupt_image") in actions
    assert ("tiny.jpg", "removed_image_all_boxes_tiny") in actions
    assert ("ok.jpg", "dropped_tiny_boxes") in actions
    assert len(read_yolo_txt(d / "labels" / "ok.txt")) == 1
    assert not (d / "images" / "tiny.jpg").exists() and not (d / "labels" / "bad.txt").exists()
    assert clean_source(d, 8, "drop_image_if_empty", EXTS) == []          # idempotent


def test_perceptual_hash_clusters_near_duplicates(tmp_path):
    make_img(tmp_path / "a.jpg", 10)
    make_img(tmp_path / "a_dup.jpg", 10, noise=3.0)
    make_img(tmp_path / "b.jpg", 11)
    make_img(tmp_path / "c.jpg", 12)
    names = ["a.jpg", "a_dup.jpg", "b.jpg", "c.jpg"]
    hs = [phash_int(tmp_path / n, 8) for n in names]
    cid = hamming_clusters(hs, 4)
    assert cid[0] == cid[1] and len({cid[0], cid[2], cid[3]}) == 3


def test_hamming_clusters_exact_small_case():
    assert hamming_clusters([0b0000, 0b0001, 0b1111, 0b1110], 1) == [0, 0, 1, 1]
    assert hamming_clusters([], 3) == []
