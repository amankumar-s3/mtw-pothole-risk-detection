"""Metrics on tiny, hand-computable examples."""
import math

import numpy as np
import pandas as pd
import pytest

from dataset_helpers import make_img, write_yolo
from ptr.eval.detection import (average_precision, best_f1_conf, evaluate_predictions, failure_gallery, iou_xyxy,
                                match_flags, prf_at_conf, run_evaluation)
from ptr.eval.fps import benchmark
from ptr.eval.risk_agreement import (agreement_report, confusion, inter_rater, load_labelled_events, split_by_clip,
                                     weighted_kappa)
from ptr.eval.tracking import evaluate_tracking, iou_distance, load_mot_gt, load_tracks_csv

A, B = (0, 0, 10, 10), (20, 20, 30, 30)
GT = {"img": [A, B]}
PRED = {"img": [(0.9, A), (0.8, (50, 50, 60, 60)), (0.7, B)]}       # TP, FP, TP (confidence order)


def test_iou():
    assert iou_xyxy(A, A) == 1.0 and iou_xyxy(A, B) == 0.0
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 15)) == pytest.approx(100 / 150)


def test_average_precision_hand_computed():
    c, t, n = match_flags(PRED, GT, 0.5)
    assert list(t) == [1, 0, 1] and n == 2
    # precision 1, 1/2, 2/3 at recall 1/2, 1/2, 1 -> envelope: 1 up to recall .5, 2/3 up to recall 1
    assert average_precision(t, n) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_prf_at_confidence_and_best_f1():
    c, t, n = match_flags(PRED, GT, 0.5)
    r = prf_at_conf(c, t, n, 0.75)               # keeps conf 0.9 (TP) and 0.8 (FP)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1) and r["f1"] == pytest.approx(0.5)
    r = prf_at_conf(c, t, n, 0.0)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == 1.0 and r["f1"] == pytest.approx(0.8)
    conf, f1 = best_f1_conf(c, t, n, 0.05, 0.05)
    assert f1 == pytest.approx(0.8) and 0.05 < conf <= 0.7      # middle of the F1 plateau [0.05, 0.7]


def test_duplicate_detection_is_a_false_positive():
    p = {"img": [(0.9, A), (0.8, A)]}
    c, t, n = match_flags(p, {"img": [A]}, 0.5)
    assert list(t) == [1, 0]


def test_predictions_in_images_without_gt_are_false_positives_and_empty_gt_gives_nan_ap():
    c, t, n = match_flags({"a": [(0.9, A)], "b": [(0.8, A)]}, {"a": [A], "b": []}, 0.5)
    assert sorted(t) == [0, 1]
    assert math.isnan(average_precision(np.array([]), 0))
    assert average_precision(np.array([]), 3) == 0.0


def test_ap_50_95_over_ten_thresholds():
    thrs = [0.5 + 0.05 * i for i in range(10)]
    r = evaluate_predictions({"i": [(0.9, (0, 0, 10, 15))]}, {"i": [(0, 0, 10, 10)]}, thrs, 0.5)
    assert r["ap50"] == 1.0 and r["ap50_95"] == pytest.approx(0.4)     # IoU 0.667: TP for 0.50, 0.55, 0.60, 0.65


def test_evaluate_predictions_counts():
    r = evaluate_predictions(PRED, GT, [0.5], 0.5)
    assert r["n_images"] == 1 and r["n_gt_boxes"] == 2 and r["conf"] == 0.5 and r["tp"] == 2 and r["fp"] == 1


def make_split(root, name, n, offset=0, boxes_per_img=1):
    for i in range(n):
        make_img(root / name / "images" / f"{name}_{i}.jpg", i + offset, size=(100, 100))
        write_yolo(root / name / "labels" / f"{name}_{i}.txt", [(0, 0.5, 0.5, 0.4, 0.4)] * boxes_per_img)


DCFG = {"iou_thresholds": [0.5, 0.75], "pred_conf_floor": 0.01, "f1_conf_grid_step": 0.05, "gallery_conf": 0.25,
        "gallery_max": 3}


def test_run_evaluation_picks_confidence_on_val_only(tmp_path):
    for s in ("val", "test_a", "test_b"):
        make_split(tmp_path, s, 6)
    GTBOX = (30.0, 30.0, 70.0, 70.0)
    JUNK = (0.0, 0.0, 10.0, 10.0)

    def predict(path):                                  # perfect box at 0.9; junk at 0.3 only in the TEST images
        out = [(0.9, GTBOX)]
        if not path.name.startswith("val"):
            out.append((0.3, JUNK))
        return out
    splits = {s: tmp_path / s for s in ("val", "test_a", "test_b")}
    res = run_evaluation("t", splits, predict, 0, [".jpg"], DCFG, tmp_path / "out", "val", None, "test_b")
    # val has no junk: F1 is flat on [0.01, 0.9] -> middle of the plateau; this is what the code must use
    assert res["conf"] == pytest.approx(0.455, abs=0.03)
    assert res["results"]["val"]["f1"] == pytest.approx(1.0)
    assert res["results"]["test_a"]["precision"] == pytest.approx(1.0)       # junk (0.3) is below the val-chosen conf
    # had the confidence been chosen on a test split, a different number would come out
    other = run_evaluation("t2", splits, predict, 0, [".jpg"], DCFG, tmp_path / "out2", "test_a", None)
    assert other["conf"] != res["conf"] and other["conf"] > 0.3
    md = (tmp_path / "out" / "detection_t.md").read_text()
    assert "F1-optimal on 'val' only" in md and "Test A vs Test B gap" in md and "WARNING" in md
    assert (tmp_path / "out" / "detection_t.csv").exists() and (tmp_path / "out" / "detection_t_val_curves.png").exists()


def test_run_evaluation_requires_a_confidence_source(tmp_path):
    make_split(tmp_path, "test_b", 2)
    with pytest.raises(ValueError):
        run_evaluation("t", {"test_b": tmp_path / "test_b"}, lambda p: [], 0, [".jpg"], DCFG, tmp_path / "o", "val", None)
    r = run_evaluation("t", {"test_b": tmp_path / "test_b"}, lambda p: [], 0, [".jpg"], DCFG, tmp_path / "o", "val", 0.3)
    assert r["results"]["test_b"]["recall"] == 0.0 and "fixed by the user" in r["note"]


def test_run_evaluation_ignores_other_gt_classes(tmp_path):
    make_img(tmp_path / "s" / "images" / "a.jpg", 1, size=(100, 100))
    write_yolo(tmp_path / "s" / "labels" / "a.txt", [(0, 0.5, 0.5, 0.4, 0.4), (1, 0.2, 0.2, 0.1, 0.1)])
    r = run_evaluation("t", {"s": tmp_path / "s"}, lambda p: [(0.9, (15.0, 15.0, 25.0, 25.0))], 1, [".jpg"], DCFG,
                       tmp_path / "o", None, 0.5)
    assert r["results"]["s"]["n_gt_boxes"] == 1 and r["results"]["s"]["f1"] == 1.0


def test_failure_gallery_orders_worst_first(tmp_path):
    for i in range(3):
        make_img(tmp_path / f"i{i}.jpg", i, size=(100, 100))
    imgs = sorted(tmp_path.glob("*.jpg"))
    gts = {"i0.jpg": [A], "i1.jpg": [A, B], "i2.jpg": [A]}
    preds = {"i0.jpg": [(0.9, A)], "i1.jpg": [], "i2.jpg": [(0.9, A)]}
    rows = failure_gallery(imgs, preds, gts, 0.25, tmp_path / "g", 5)
    assert [r["source_image"] for r in rows] == ["i1.jpg"] and rows[0]["false_negatives"] == 2
    assert (tmp_path / "g" / rows[0]["file"]).exists()


# ------------------------------------------------------------------------------------------ tracking
def mot(rows):
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h"])


G = [[f, i, 0 if i == 1 else 50, 0 if i == 1 else 50, 10, 10] for f in (1, 2, 3) for i in (1, 2)]


def test_tracking_perfect():
    h = mot([[f, 10 + i, x, y, w, hh] for f, i, x, y, w, hh in G])
    r = evaluate_tracking(mot(G), h, 0.5)
    assert r["id_switches"] == 0 and r["idf1"] == pytest.approx(1.0) and r["mota"] == pytest.approx(1.0)
    assert r["motp_mean_iou"] == pytest.approx(1.0)


def test_tracking_id_switches_hand_computed():
    hyp = []
    for f, i, x, y, w, hh in G:
        hid = 10 + i if f < 3 else 10 + (3 - i)              # ids swapped in the last frame
        hyp.append([f, hid, x, y, w, hh])
    r = evaluate_tracking(mot(G), mot(hyp), 0.5)
    assert r["id_switches"] == 2 and r["num_gt_boxes"] == 6
    assert r["mota"] == pytest.approx(1 - 2 / 6) and r["idf1"] == pytest.approx(4 / 6)


def test_tracking_misses_and_false_positives():
    hyp = mot([[1, 7, 0, 0, 10, 10], [1, 8, 200, 200, 10, 10]])      # 1 match, 1 FP, 5 GT boxes missed
    r = evaluate_tracking(mot(G), hyp, 0.5)
    assert r["misses"] == 5 and r["false_positives"] == 1 and r["id_switches"] == 0


def test_iou_distance_threshold():
    d = iou_distance(np.array([[0, 0, 10, 10.0]]), np.array([[0, 0, 10, 10.0], [2.5, 0, 10, 10.0], [5, 0, 10, 10.0], [100, 0, 10, 10.0]]), 0.5)
    # shifted by 2.5: intersection 75, union 125 -> IoU 0.6, distance 0.4; shifted by 5: IoU 1/3 -> beyond 0.5 -> NaN
    assert d[0, 0] == 0.0 and d[0, 1] == pytest.approx(0.4) and math.isnan(d[0, 2]) and math.isnan(d[0, 3])


def test_mot_loading_and_frame_mapping(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text("1,1,10.5,20,30,40,1,-1,-1\n2, 1, 11, 21, 30, 40, 1, -1, -1\n")
    g = load_mot_gt(p)
    assert list(g["frame"]) == [1, 2] and g["x"][0] == 10.5
    pd.DataFrame({"frame": [0, 1], "track_id": [3, 3], "x1": [10.0, 11.0], "y1": [20.0, 21.0], "x2": [40.0, 41.0],
                  "y2": [60.0, 61.0]}).to_csv(tmp_path / "tracks.csv", index=False)
    h = load_tracks_csv(tmp_path / "tracks.csv", frame_stride=2, frame_offset=1)
    assert list(h["frame"]) == [1, 3] and list(h["w"]) == [30.0, 30.0]


# ------------------------------------------------------------------------------------------ risk agreement
def test_confusion_accuracy_and_weighted_kappa_hand_computed():
    ref = ["Low", "Low", "Moderate", "High"]
    sysl = ["Low", "Moderate", "Moderate", "High"]
    assert confusion(ref, sysl).tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    rep = agreement_report(ref, sysl)
    assert rep["n"] == 4 and rep["accuracy"] == pytest.approx(0.75)
    assert rep["weighted_kappa"] == pytest.approx(1 - 0.125 / 0.4375)           # 0.7143, worked out by hand
    assert rep["per_class"]["Low"]["precision"] == 1.0 and rep["per_class"]["Low"]["recall"] == 0.5
    assert rep["per_class"]["Moderate"]["precision"] == 0.5 and rep["per_class"]["Moderate"]["recall"] == 1.0


def test_kappa_edge_cases():
    assert weighted_kappa(["Low", "Moderate", "High"], ["Low", "Moderate", "High"]) == pytest.approx(1.0)
    assert math.isnan(weighted_kappa(["Low"] * 3, ["Low"] * 3))                   # undefined, must not crash
    assert math.isnan(weighted_kappa([], []))
    with pytest.raises(ValueError):
        confusion(["Severe"], ["Low"])
    ir = inter_rater(["Low", "High", "Moderate"], ["low", "High", "High"])        # case-insensitive
    assert ir["n"] == 3 and ir["exact_agreement"] == pytest.approx(2 / 3)


def test_load_labelled_events_drops_incomplete(tmp_path):
    pd.DataFrame({"event_id": ["E1", "E2", "E3"], "clip_id": ["c", "c", "d"], "peak_class": ["Low", "High", "Low"]}).to_csv(tmp_path / "e.csv", index=False)
    pd.DataFrame({"event_id": ["E1", "E2", "E3"], "clip_id": ["c", "c", "d"], "rater1": ["Low", "", "Low"],
                  "rater2": ["Low", "High", None]}).to_csv(tmp_path / "l.csv", index=False)
    df = load_labelled_events(tmp_path / "e.csv", tmp_path / "l.csv")
    assert list(df["event_id"]) == ["E1"]


def test_split_by_clip_is_disjoint_deterministic_and_never_empty():
    clips = ["a", "a", "b", "c", "c", "c", "d"]
    cal, held = split_by_clip(clips, 0.5, 3)
    assert not set(cal) & set(held) and set(cal) | set(held) == {"a", "b", "c", "d"}
    assert (cal, held) == split_by_clip(clips, 0.5, 3)
    cal, held = split_by_clip(["x", "y"], 0.0, 0)
    assert len(cal) == 1 and len(held) == 1


def test_benchmark_excludes_warmup():
    calls = []
    r = benchmark(lambda x: calls.append(x), list(range(12)), warmup=2)
    assert len(calls) == 12 and r["n_timed"] == 10 and r["warmup"] == 2 and r["fps_mean"] > 0
    with pytest.raises(ValueError):
        benchmark(lambda x: x, [1, 2], warmup=2)
