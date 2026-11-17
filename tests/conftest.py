from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # scripts are importable as modules in tests


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture()
def pipe_cfg(root):
    from ptr.config import load_pipeline_config

    return load_pipeline_config(root / "configs" / "pipeline.yaml")


@pytest.fixture()
def risk_cfg(root):
    from ptr.config import load_risk_config

    return load_risk_config(root / "configs" / "risk.yaml")


@pytest.fixture(scope="session")
def mock_run(tmp_path_factory, root):
    """Synthetic demo video + mock detections, processed once by the real Pipeline (no weights needed)."""
    from ptr.config import load_pipeline_config, load_risk_config
    from ptr.detection.detector import MockDetections
    from ptr.pipeline import Pipeline
    from ptr.synthetic import demo_scene, export_mock, render_video, shake_sequence, simulate

    d = tmp_path_factory.mktemp("mock")
    scene = demo_scene(duration=8.0, seed=0)
    shake = shake_sequence(scene.n_frames, 4.0, seed=1)
    truth = simulate(scene, shake=shake)
    video, mock = d / "demo.mp4", d / "demo.json"
    render_video(scene, video, shake, truth)
    export_mock(truth, scene, mock)
    cfg = load_pipeline_config(root / "configs" / "pipeline.yaml")
    risk = load_risk_config(root / "configs" / "risk.yaml")
    src = MockDetections.from_file(mock, cfg.detectors.pothole.min_box_px, cfg.detectors.mtw.min_box_px)
    pipe = Pipeline(cfg, risk, src, out_root=d / "out", run_id="smoke", meta={"mock_detections": str(mock)})
    summary = pipe.run(video)
    return {"dir": pipe.run_dir, "summary": summary, "video": video, "mock": mock, "cfg": cfg, "risk": risk,
            "scene": scene}
