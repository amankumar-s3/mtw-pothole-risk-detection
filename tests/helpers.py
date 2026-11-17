"""Shared test helpers: run the feature extractor on synthetic scenes."""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import numpy as np

from ptr.analysis.egomotion import EgoHistory
from ptr.analysis.features import FeatureExtractor
from ptr.schemas import InteractionFeatures, PotholeObs, Track
from ptr.synthetic import Scene, shake_to_ego, simulate


def run_features(scene: Scene, pipe_cfg, shake: Optional[np.ndarray] = None, use_true_ego: bool = False,
                 ego_compensation: bool = True, hide_pothole: Optional[range] = None,
                 features_overrides: Optional[dict] = None, noisy: bool = False) -> list[InteractionFeatures]:
    fcfg = dataclasses.replace(pipe_cfg.features, ego_compensation=ego_compensation, **(features_overrides or {}))
    truth = simulate(scene, shake=shake, noisy=noisy)
    n = len(truth)
    hist = EgoHistory(int(math.ceil(pipe_cfg.ego_motion.max_span_seconds * scene.fps)))
    egos = shake_to_ego(shake if shake is not None else np.zeros((n, 2)))
    fe = FeatureExtractor(fcfg, scene.fps, (scene.cam.width, scene.cam.height),
                          pipe_cfg.trajectory.min_history_frames, hist)
    rows: list[InteractionFeatures] = []
    ages: dict[int, int] = {}
    for ft in truth:
        M = egos[ft.frame] if use_true_ego else np.array([[1.0, 0, 0], [0, 1.0, 0]])
        hist.push(ft.frame, M, True)
        tracks = []
        for mid, b in ft.mtws.items():
            ages[mid] = ages.get(mid, 0) + 1
            tracks.append(Track(mid, ft.frame, ft.t, b, b, 0.9, ((b[0] + b[2]) / 2, b[3]), age_frames=ages[mid],
                                reliable=True))
        poth = []
        if not (hide_pothole is not None and ft.frame in hide_pothole):
            for pid, b in ft.potholes.items():
                poth.append(PotholeObs(pid, ft.frame, ft.t, b, 0.9, True, False, ft.frame, 5, "small"))
        rows.extend(fe.update(ft.frame, ft.t, tracks, poth))
    return rows


def encounter_row(rows: list[InteractionFeatures]) -> Optional[InteractionFeatures]:
    for r in rows:
        if r.encounter:
            return r
    return None


def max_finite(vals) -> float:
    v = [x for x in vals if math.isfinite(x)]
    return max(v) if v else math.nan
