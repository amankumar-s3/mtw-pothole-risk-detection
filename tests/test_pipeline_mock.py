"""End-to-end smoke test in mock mode: synthetic video + mock detections, no weights, no GPU."""
import csv
import json

import cv2
import yaml

from ptr.schemas import RISK_LABELS

FILES = ["annotated.mp4", "tracks.csv", "potholes.csv", "interactions.csv", "events.csv", "summary.json",
         "egomotion.csv", "config_used.yaml", "run.log"]


def read_csv(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def test_all_outputs_exist(mock_run):
    for f in FILES:
        p = mock_run["dir"] / f
        assert p.is_file() and p.stat().st_size > 0, f


def test_summary_is_consistent_with_events(mock_run):
    s = mock_run["summary"]
    ev = read_csv(mock_run["dir"] / "events.csv")
    assert s["n_events"] == len(ev) > 0
    assert sum(s["events_per_class"].values()) == len(ev)
    for e in ev:
        assert e["peak_class"] in RISK_LABELS and len(e["reason"]) > 10
    assert "NOT accident prediction" in s["status"]
    assert s["mock_detections"] is True and s["camera_mode"] == "moving"
    assert json.loads((mock_run["dir"] / "summary.json").read_text())["frames_processed"] == s["frames_processed"]


def test_annotated_video_matches_input(mock_run):
    cap = cv2.VideoCapture(str(mock_run["dir"] / "annotated.mp4"))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n == mock_run["summary"]["frames_processed"] == mock_run["scene"].n_frames
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (960, 540)


def test_config_used_is_saved_and_labelled(mock_run):
    d = yaml.safe_load((mock_run["dir"] / "config_used.yaml").read_text())
    assert d["risk"]["status"] == "initial hypotheses, to be calibrated"
    assert d["run"]["camera_mode"] == "moving" and d["pipeline"]["ego_motion"]["enabled"] is True
    assert d["risk_block_used"]["weights"]["decel"] == 0.0


def test_run_log_has_run_id(mock_run):
    assert "smoke" in (mock_run["dir"] / "run.log").read_text()


def test_csv_schemas(mock_run):
    d = mock_run["dir"]
    assert {"frame", "t", "track_id", "x1", "y2", "contact_x", "reliable"} <= set(read_csv(d / "tracks.csv")[0])
    assert {"pothole_id", "confirmed", "last_known", "size_class"} <= set(read_csv(d / "potholes.csv")[0])
    inter = read_csv(d / "interactions.csv")
    assert {"dy", "dx", "proximity", "ttr", "swerve", "decel", "clearance", "total", "accepted_label",
            "reason"} <= set(inter[0])
    assert {"valid", "inlier_ratio", "tx", "ty"} <= set(read_csv(d / "egomotion.csv")[0])
    assert len(read_csv(d / "egomotion.csv")) == mock_run["summary"]["frames_processed"]


def test_ego_motion_runs_and_is_mostly_valid(mock_run):
    s = mock_run["summary"]
    assert s["ego_motion_enabled"] and s["ego_motion_valid_fraction"] > 0.8


def test_one_row_per_track_per_frame(mock_run):
    rows = read_csv(mock_run["dir"] / "tracks.csv")
    keys = [(r["track_id"], r["frame"]) for r in rows]
    assert len(keys) == len(set(keys)) and {r["track_id"] for r in rows} == {"1", "2", "3"}


def test_synthetic_behaviours_are_recovered(mock_run):
    """MTW 2 rides through a pothole, MTW 3 swerves late, MTW 1 passes wide (synthetic ground truth)."""
    ev = read_csv(mock_run["dir"] / "events.csv")
    by = lambda m: [e for e in ev if e["mtw_id"] == str(m)]
    through = [e for e in by(2) if e["encountered"] == "True" and float(e["clearance"]) < 0.5]
    assert through, "MTW 2 should have an encounter with ~0 clearance"
    assert any(e["peak_class"] == "High" for e in by(3)), "late swerve should reach High"
    assert not any(e["peak_class"] == "High" for e in by(1)), "wide pass must not be High"


def test_reproducible_run(mock_run, tmp_path):
    from ptr.detection.detector import MockDetections
    from ptr.pipeline import Pipeline
    cfg, risk = mock_run["cfg"], mock_run["risk"]
    src = MockDetections.from_file(mock_run["mock"], cfg.detectors.pothole.min_box_px, cfg.detectors.mtw.min_box_px)
    p2 = Pipeline(cfg, risk, src, out_root=tmp_path, run_id="again", meta={"mock_detections": "x"})
    p2.run(mock_run["video"])
    assert (p2.run_dir / "events.csv").read_text() == (mock_run["dir"] / "events.csv").read_text()


def test_cli_static_mode_uses_static_block(mock_run, tmp_path, root, monkeypatch):
    monkeypatch.chdir(root)
    import run_pipeline
    rc = run_pipeline.main(["--video", str(mock_run["video"]), "--mock-detections", str(mock_run["mock"]),
                            "--out", str(tmp_path), "--run-id", "static", "--camera-mode", "static",
                            "--max-frames", "90", "--no-video"])
    assert rc == 0
    d = yaml.safe_load((tmp_path / "static" / "config_used.yaml").read_text())
    assert d["run"]["camera_mode"] == "static" and d["risk_block_used"]["weights"]["decel"] > 0
    assert not (tmp_path / "static" / "annotated.mp4").exists()
    assert json.loads((tmp_path / "static" / "summary.json").read_text())["frames_processed"] == 90


def test_make_synthetic_video_script(tmp_path):
    import make_synthetic_video
    out = tmp_path / "s" / "demo"
    assert make_synthetic_video.main(["--out", str(out), "--duration", "1.0", "--width", "320", "--height", "180"]) == 0
    assert out.with_suffix(".mp4").stat().st_size > 0
    doc = json.loads(out.with_suffix(".json").read_text())
    assert len(doc["frames"]) == 30 and doc["source"] == "synthetic"
