#!/usr/bin/env python
"""Create a short synthetic dashcam-style video + matching mock detections (no weights, no downloads).

Scene (made-up geometry, see src/ptr/synthetic.py): moving camera with shake, three potholes, three MTWs
moving away: one passes far to the side, one rides THROUGH its pothole, one swerves late before its pothole.

  python scripts/make_synthetic_video.py --out data/videos/synthetic_demo
  -> data/videos/synthetic_demo.mp4, synthetic_demo.json (mock detections), synthetic_demo.scene.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ptr.synthetic import CameraModel, demo_scene, export_mock, render_video, shake_sequence, simulate
from ptr.utils import seed_everything


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/videos/synthetic_demo", help="path prefix (no extension)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=8.0, help="seconds")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--shake-px", type=float, default=4.0, help="camera shake amplitude in pixels (0 = none)")
    ap.add_argument("--noise-px", type=float, default=1.0, help="box jitter of the mock detections")
    ap.add_argument("--dropout", type=float, default=0.08, help="probability a pothole detection is missing")
    a = ap.parse_args(argv)

    seed_everything(a.seed)
    cam = CameraModel(width=a.width, height=a.height, f=500.0 * a.width / 640.0,
                      horizon_row=150.0 * a.height / 360.0)
    scene = demo_scene(fps=a.fps, duration=a.duration, seed=a.seed, noise_px=a.noise_px, dropout=a.dropout, cam=cam)
    shake = shake_sequence(scene.n_frames, a.shake_px, seed=a.seed + 1)
    truth = simulate(scene, shake=shake)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_video(scene, out.with_suffix(".mp4"), shake, truth)
    export_mock(truth, scene, out.with_suffix(".json"))
    out.with_suffix(".scene.json").write_text(json.dumps({
        "note": "synthetic test scene; the numbers are made up and not measurements",
        "mtws": [{"mtw_id": m.mtw_id, "scenario": m.scenario} for m in scene.mtws],
        "frames": scene.n_frames, "fps": scene.fps, "shake_px": a.shake_px}, indent=2))
    print(f"wrote {out.with_suffix('.mp4')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
