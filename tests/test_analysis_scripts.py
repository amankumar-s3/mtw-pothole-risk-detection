"""Labeling / analysis scripts run on the mock end-to-end run (no weights needed)."""
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import evaluate_risk_agreement
import evaluate_tracking
import export_encounters_for_labeling
import make_figures
import sensitivity_analysis
import split_events_by_clip
from ptr.analysis.rescore import rescore
from ptr.synthetic import shake_sequence, simulate

ROOT = Path(__file__).resolve().parents[1]
DP = str(ROOT / "configs" / "data_prep.yaml")


def test_rescore_reproduces_the_pipeline_events(mock_run):
    df = pd.read_csv(mock_run["dir"] / "interactions.csv")
    ev = pd.read_csv(mock_run["dir"] / "events.csv")
    block = mock_run["risk"].for_mode("moving")
    again = rescore(df, block, mock_run["summary"]["effective_fps"])
    assert len(again) == len(ev)
    got = sorted((e.mtw_id, e.pothole_id, e.start_frame, e.end_frame, e.peak_class) for e in again)
    exp = sorted(zip(ev.mtw_id, ev.pothole_id, ev.start_frame, ev.end_frame, ev.peak_class))
    assert got == [tuple(map(lambda v: v.item() if hasattr(v, "item") else v, r)) for r in exp]


def test_sensitivity_script(mock_run, tmp_path):
    rc = sensitivity_analysis.main(["--run", str(mock_run["dir"]), "--risk-config", str(ROOT / "configs" / "risk.yaml"),
                                    "--data-prep-config", DP, "--out-dir", str(tmp_path), "--name", "sens"])
    assert rc == 0
    df = pd.read_csv(tmp_path / "sens.csv")
    assert {"parameter", "factor", "events", "class_up", "class_down", "event_lost", "changed"} <= set(df.columns)
    assert set(df["factor"]) == {0.8, 1.2} and (df["events"] == mock_run["summary"]["n_events"]).all()
    assert (df["changed"] == df.class_up + df.class_down + df.event_lost).all()
    assert df["changed"].max() > 0                      # at least one perturbation changes something
    assert "initial hypotheses" in (tmp_path / "sens.md").read_text()
    assert not any(df["parameter"].str.contains("weight.decel"))       # weight 0 in moving mode: not perturbed


def test_sensitivity_rejects_mixed_modes(mock_run, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(mock_run["dir"], other, ignore=shutil.ignore_patterns("*.mp4"))
    s = json.loads((other / "summary.json").read_text())
    s["camera_mode"] = "static"
    (other / "summary.json").write_text(json.dumps(s))
    assert sensitivity_analysis.main(["--run", str(mock_run["dir"]), "--run", str(other), "--data-prep-config", DP,
                                      "--out-dir", str(tmp_path / "o")]) == 2


def test_export_encounters_blind_and_complete(mock_run, tmp_path):
    out = tmp_path / "lab"
    assert export_encounters_for_labeling.main(["--run", str(mock_run["dir"]), "--out", str(out),
                                                "--data-prep-config", DP]) == 0
    ev = pd.read_csv(mock_run["dir"] / "events.csv")
    rows = list(csv.DictReader(open(out / "labels_template.csv")))
    assert len(rows) == len(ev) and list(rows[0]) == ["event_id", "clip_id", "rater1", "rater2", "notes"]
    assert all(r["rater1"] == "" and r["rater2"] == "" for r in rows)
    text = (out / "labels_template.csv").read_text()
    assert not any(w in text for w in ("Low", "Moderate", "High"))      # blind: no system output in the template
    clips = sorted((out / "clips").glob("*.mp4"))
    assert len(clips) == len(ev) and all(c.name.startswith("demo__E") for c in clips)
    cap = cv2.VideoCapture(str(clips[0]))
    ok, frame = cap.read()
    assert ok and frame.shape[:2] == (540, 960)
    assert not any(w in c.name for c in clips for w in ("Low", "Moderate", "High"))


def test_split_events_by_clip_script(mock_run, tmp_path):
    ev = pd.read_csv(mock_run["dir"] / "events.csv")
    parts = []
    for i, clip in enumerate(["c1", "c2", "c3", "c4"]):
        d = ev.copy()
        d["clip_id"] = clip
        d.to_csv(tmp_path / f"e{i}.csv", index=False)
        parts.append(str(tmp_path / f"e{i}.csv"))
    assert split_events_by_clip.main(["--events", *parts, "--out", str(tmp_path / "o"), "--data-prep-config", DP]) == 0
    cal = pd.read_csv(tmp_path / "o" / "events_calibration.csv")
    held = pd.read_csv(tmp_path / "o" / "events_heldout.csv")
    assert not set(cal.clip_id) & set(held.clip_id) and len(cal) + len(held) == 4 * len(ev)
    assert len(set(cal.clip_id)) == 2 and len(set(held.clip_id)) == 2


def test_make_figures_script(mock_run, tmp_path):
    assert make_figures.main(["--run", str(mock_run["dir"]), "--out", str(tmp_path)]) == 0
    for f in ("risk_timeline.png", "class_counts.png", "pothole_positions.png"):
        img = cv2.imread(str(tmp_path / f))
        assert img is not None and img.size > 0


def test_evaluate_tracking_against_synthetic_truth(mock_run, tmp_path):
    """The mock detections are the synthetic truth (+1 px noise, stable ids): near-perfect tracking expected."""
    scene = mock_run["scene"]
    truth = simulate(scene, shake=shake_sequence(scene.n_frames, 4.0, seed=1), noisy=False)
    lines = []
    for ft in truth:
        for mid, (x1, y1, x2, y2) in ft.mtws.items():
            lines.append(f"{ft.frame + 1},{mid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1")
    gt = tmp_path / "gt.txt"
    gt.write_text("\n".join(lines))
    assert evaluate_tracking.main(["--gt", str(gt), "--run", str(mock_run["dir"]), "--out-dir", str(tmp_path / "o"),
                                   "--data-prep-config", DP]) == 0
    res = json.loads((tmp_path / "o" / "tracking_smoke.json").read_text())
    assert res["id_switches"] == 0 and res["idf1"] > 0.95 and res["num_gt_boxes"] > 100
    assert res["n_gt_tracks"] == 3 and res["n_hyp_tracks"] == 3


def test_evaluate_risk_agreement_script(mock_run, tmp_path):
    ev = pd.read_csv(mock_run["dir"] / "events.csv")
    lab = pd.DataFrame({"event_id": ev.event_id, "clip_id": ev.clip_id, "rater1": ev.peak_class, "rater2": ev.peak_class,
                        "notes": ""})
    lab.to_csv(tmp_path / "labels.csv", index=False)
    assert evaluate_risk_agreement.main(["--events", str(mock_run["dir"] / "events.csv"), "--labels", str(tmp_path / "labels.csv"),
                                         "--out-dir", str(tmp_path / "o"), "--name", "t"]) == 0
    rep = json.loads((tmp_path / "o" / "risk_agreement_t.json").read_text())
    assert rep["n"] == len(ev) and rep["accuracy"] == 1.0 and rep["inter_rater"]["exact_agreement"] == 1.0
    md = (tmp_path / "o" / "risk_agreement_t.md").read_text()
    assert f"(n): **{len(ev)}**" in md and "WARNING" in md        # n < 30 => exploratory warning
    lab.loc[:, ["rater1", "rater2"]] = ""
    lab.to_csv(tmp_path / "empty.csv", index=False)
    assert evaluate_risk_agreement.main(["--events", str(mock_run["dir"] / "events.csv"), "--labels", str(tmp_path / "empty.csv"),
                                         "--out-dir", str(tmp_path / "o")]) == 2


def test_rescore_run_script_matches_and_follows_yaml_changes(mock_run, tmp_path):
    import rescore_run
    import yaml
    out = tmp_path / "ev.csv"
    assert rescore_run.main(["--run", str(mock_run["dir"]), "--risk-config", str(ROOT / "configs" / "risk.yaml"),
                             "--out", str(out)]) == 0
    base = pd.read_csv(mock_run["dir"] / "events.csv")
    again = pd.read_csv(out)
    assert list(again["event_id"]) == list(base["event_id"]) and list(again["peak_class"]) == list(base["peak_class"])
    cfg = yaml.safe_load((ROOT / "configs" / "risk.yaml").read_text())
    cfg["moving"]["classes"] = {"low_max": 0.9, "high_min": 0.95}               # everything becomes Low
    p = tmp_path / "risk_changed.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out2 = tmp_path / "ev2.csv"
    assert rescore_run.main(["--run", str(mock_run["dir"]), "--risk-config", str(p), "--out", str(out2)]) == 0
    changed = pd.read_csv(out2)
    assert list(changed["event_id"]) == list(base["event_id"])                    # same encounters, same ids (labels stay valid)
    assert (changed["peak_class"] != base["peak_class"]).any()
