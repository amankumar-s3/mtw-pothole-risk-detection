import dataclasses
import math

import numpy as np
import pytest

from helpers import encounter_row, max_finite, run_features
from ptr.synthetic import (MtwSpec, Scene, scenario, shake_sequence, shake_to_ego, simulate)


def rows_for(name, direction, pipe_cfg, **kw):
    return run_features(scenario(name, direction), pipe_cfg, **kw)


def test_far_pass_has_large_clearance(pipe_cfg):
    rows = rows_for("far_pass", "away", pipe_cfg)
    e = encounter_row(rows)
    assert e is not None and e.direction == "away"
    assert e.clearance > 3.0 and e.clearance_from_crossing


def test_pass_through_pothole_has_zero_clearance(pipe_cfg):
    rows = rows_for("through", "away", pipe_cfg)
    e = encounter_row(rows)
    assert e is not None
    assert e.clearance < 0.2
    assert min(r.proximity for r in rows if not r.insufficient_history) < 0.5


def test_late_swerve_is_detected_and_straight_is_not(pipe_cfg):
    sw = max_finite(r.swerve for r in rows_for("late_swerve", "away", pipe_cfg))
    st = max_finite(r.swerve for r in rows_for("through", "away", pipe_cfg))
    assert sw > 1.0
    assert st < 0.35                                   # constant approach: no false swerve
    assert sw > 4 * st


def test_early_swerve_is_outside_the_pair_window_or_small_near_encounter(pipe_cfg):
    rows = rows_for("early_swerve", "away", pipe_cfg)
    near = [r for r in rows if abs(r.t - 2.1) < 0.2 and math.isfinite(r.swerve)]
    assert near and max(r.swerve for r in near) < 0.35


def test_constant_approach_registers_no_deceleration(pipe_cfg):
    rows = rows_for("through", "away", pipe_cfg)
    d = [r.decel for r in rows if math.isfinite(r.decel)]
    assert d and max(abs(x) for x in d) < 0.05


def test_braking_registers_deceleration_in_static_scene(pipe_cfg):
    """Static camera (cam_speed 0): the MTW slows from 9 to 2 m/s around t = 1.5 s."""
    sc = scenario("through", "away", cam_speed=0.0)
    z0 = sc.mtws[0].Z0
    def Z(t):
        if t < 1.3:
            return z0 + 9.0 * t
        return z0 + 9.0 * 1.3 + 2.0 * (t - 1.3)
    sc.mtws[0] = MtwSpec(1, z0, 9.0, sc.mtws[0].X_fn, "braking", Z_fn=Z)
    rows = run_features(sc, pipe_cfg, features_overrides={"max_abs_dy": 10.0})
    assert max_finite(r.decel for r in rows) > 0.4


def test_standstill_is_flagged_and_decel_undefined(pipe_cfg):
    sc = scenario("through", "away", cam_speed=0.0)
    sc.mtws[0] = MtwSpec(1, 30.0, 0.0, sc.mtws[0].X_fn, "still")
    rows = run_features(sc, pipe_cfg, features_overrides={"max_abs_dy": 10.0})
    ok = [r for r in rows if not r.insufficient_history]
    assert ok and all(r.standstill and math.isnan(r.decel) for r in ok)


def test_insufficient_history_is_flagged(pipe_cfg):
    rows = rows_for("through", "away", pipe_cfg)
    assert rows[0].insufficient_history and math.isnan(rows[0].swerve)
    assert not rows[-1].insufficient_history


@pytest.mark.parametrize("name", ["far_pass", "through", "late_swerve"])
@pytest.mark.parametrize("use_ego", [False, True])
def test_shared_camera_shake_leaves_features_unchanged(pipe_cfg, name, use_ego):
    sc = scenario(name, "away")
    clean = run_features(sc, pipe_cfg)
    shake = shake_sequence(sc.n_frames, 8.0, seed=3)
    shaky = run_features(sc, pipe_cfg, shake=shake, use_true_ego=use_ego)
    assert len(clean) == len(shaky)
    for a, b in zip(clean, shaky):
        assert a.frame == b.frame
        for fld in ("dy", "dx", "proximity", "swerve", "clearance"):
            x, y = getattr(a, fld), getattr(b, fld)
            if math.isfinite(x) or math.isfinite(y):
                assert math.isclose(x, y, abs_tol=1e-6), (fld, a.frame, x, y)


def _mirror(scene: Scene) -> Scene:
    import copy
    m = copy.deepcopy(scene)
    for p in m.potholes:
        p.X = -p.X
    m.mtws = [MtwSpec(s.mtw_id, s.Z0, s.speed, (lambda f: (lambda t: -f(t)))(s.X_fn), s.scenario, s.Z_fn)
              for s in scene.mtws]
    return m


@pytest.mark.parametrize("name", ["through", "near_pass", "late_swerve"])
def test_left_right_symmetry(pipe_cfg, name):
    sc = scenario(name, "away", pothole_X=0.6)
    a = run_features(sc, pipe_cfg)
    b = run_features(_mirror(sc), pipe_cfg)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert math.isclose(x.dx, y.dx, abs_tol=1e-6)
        assert math.isclose(x.dy, y.dy, abs_tol=1e-6)
        if math.isfinite(x.swerve):
            assert math.isclose(x.swerve, y.swerve, abs_tol=1e-6)
            assert math.isclose(x.swerve_signed, -y.swerve_signed, abs_tol=1e-6)


@pytest.mark.parametrize("direction", ["away", "oncoming"])
def test_both_travel_directions(pipe_cfg, direction):
    through = rows_for("through", direction, pipe_cfg)
    far = rows_for("far_pass", direction, pipe_cfg)
    swerve = rows_for("late_swerve", direction, pipe_cfg)
    e = encounter_row(through)
    assert e is not None and e.direction == direction and e.clearance < 0.2
    assert encounter_row(far).clearance > 3.0
    assert max_finite(r.swerve for r in swerve) > 1.0
    assert max_finite(r.swerve for r in through) < 0.35


def test_dy_sign_sequences(pipe_cfg):
    away = [r.dy for r in rows_for("through", "away", pipe_cfg)]
    onc = [r.dy for r in rows_for("through", "oncoming", pipe_cfg)]
    assert away[0] > 0 > away[-1]
    assert onc[0] < 0 < onc[-1]


def test_rows_stop_after_encounter_window(pipe_cfg):
    rows = rows_for("through", "away", pipe_cfg)
    e = encounter_row(rows)
    last = max(r.frame for r in rows)
    assert last - e.frame <= math.ceil(pipe_cfg.features.post_encounter_seconds * 30) + 1


def test_size_class_uses_width_ratio_near_the_encounter(pipe_cfg):
    rows = rows_for("through", "away", pipe_cfg)
    e = encounter_row(rows)
    assert e.size_source == "ratio" and e.size_class == "large"      # 0.9 m pothole vs 0.7 m MTW


def test_off_axis_pothole_false_swerve_and_perspective_option(pipe_cfg):
    """Documented limitation: without depth matching, an off-axis pothole makes a straight rider look like
    he drifts laterally. The optional perspective correction removes most of it."""
    sc = scenario("through", "away", pothole_X=2.5)
    raw = max_finite(r.swerve for r in run_features(sc, pipe_cfg))
    persp = {"perspective": dataclasses.replace(pipe_cfg.features.perspective, enabled=True,
                                                horizon_row_frac=150 / 360)}
    cor = max_finite(r.swerve for r in run_features(sc, pipe_cfg, features_overrides=persp))
    assert raw > 0.5
    assert cor < 0.15 and cor < raw / 3


def test_last_known_pothole_benefits_from_ego_compensation(pipe_cfg):
    sc = scenario("through", "away")
    shake = shake_sequence(sc.n_frames, 8.0, seed=5)
    hide = range(55, 75)
    ref = {r.frame: r for r in run_features(sc, pipe_cfg, shake=shake)}
    with_ego = {r.frame: r for r in run_features(sc, pipe_cfg, shake=shake, use_true_ego=True, hide_pothole=hide)}
    no_ego = {r.frame: r for r in run_features(sc, pipe_cfg, shake=shake, ego_compensation=False,
                                               hide_pothole=hide)}
    frames = [f for f in hide if f in ref and f in with_ego and f in no_ego]
    assert frames
    assert all(with_ego[f].pothole_last_known for f in frames)
    e1 = np.mean([abs(with_ego[f].dy - ref[f].dy) for f in frames])
    e0 = np.mean([abs(no_ego[f].dy - ref[f].dy) for f in frames])
    assert e1 < e0


def test_invalid_ego_is_flagged(pipe_cfg):
    sc = scenario("through", "away")
    from ptr.analysis.egomotion import EgoHistory
    from ptr.analysis.features import FeatureExtractor
    from ptr.schemas import PotholeObs, Track
    hist = EgoHistory(90)
    fe = FeatureExtractor(pipe_cfg.features, 30.0, (640, 360), 5, hist)
    truth = simulate(sc, noisy=False)
    flagged = False
    ages = 0
    for ft in truth:
        hist.push(ft.frame, np.array([[1.0, 0, 0], [0, 1.0, 0]]), ft.frame % 5 != 0)   # every 5th invalid
        ages += 1
        tr = [Track(1, ft.frame, ft.t, b, b, 0.9, (0, 0), age_frames=ages) for b in ft.mtws.values()]
        po = [PotholeObs(1, ft.frame, ft.t, b, 0.9, True, False, ft.frame, 5, "small") for b in ft.potholes.values()]
        flagged |= any(r.ego_invalid for r in fe.update(ft.frame, ft.t, tr, po))
    assert flagged
