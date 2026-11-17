import math

import cv2
import numpy as np

from ptr.analysis.egomotion import (EgoHistory, EgoMotionEstimator, decompose, drop_scale, to3)


def textured(h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def warp(img, M):
    return cv2.warpAffine(img, np.asarray(M, np.float32), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_recovers_translation(pipe_cfg):
    est = EgoMotionEstimator(pipe_cfg.ego_motion)
    a = textured()
    b = warp(a, [[1, 0, 4.0], [0, 1, -3.0]])
    r = est.estimate(1, a, b, [])
    assert r.valid and r.reason == "ok"
    assert abs(r.tx - 4.0) < 0.4 and abs(r.ty + 3.0) < 0.4
    assert r.inlier_ratio > 0.8


def test_recovers_small_rotation(pipe_cfg):
    est = EgoMotionEstimator(pipe_cfg.ego_motion)
    a = textured(seed=1)
    R = cv2.getRotationMatrix2D((320, 180), 1.0, 1.0)
    r = est.estimate(1, a, warp(a, R), [])
    assert r.valid and abs(abs(r.rot_deg) - 1.0) < 0.3


def test_low_texture_falls_back_to_identity(pipe_cfg):
    est = EgoMotionEstimator(pipe_cfg.ego_motion)
    flat = np.full((360, 640), 120, np.uint8)
    r = est.estimate(1, flat, flat, [])
    assert not r.valid and r.reason in ("few_points", "few_tracked")
    assert np.allclose(r.M, [[1, 0, 0], [0, 1, 0]])


def test_boxes_are_masked_out(pipe_cfg):
    """A textured patch moving differently INSIDE a detected box must not bias the estimate."""
    est = EgoMotionEstimator(pipe_cfg.ego_motion)
    a = textured(seed=2)
    b = warp(a, [[1, 0, 2.0], [0, 1, 0.0]])
    patch = textured(60, 120, seed=9)
    ax, ay = 260, 150                                    # inside the ROI band (rows 108..252)
    a2, b2 = a.copy(), b.copy()
    a2[ay:ay + 60, ax:ax + 120] = patch
    b2[ay:ay + 60, ax + 12:ax + 132] = patch             # the "object" moves 12 px, background 2 px
    box = (ax, ay, ax + 120, ay + 60)
    r = est.estimate(1, a2, b2, [(box[0] + 12, box[1], box[2] + 12, box[3]), box])
    assert r.valid and abs(r.tx - 2.0) < 0.5


def test_no_scale_option_keeps_centre_motion():
    M = np.array([[1.02, 0.0, 3.0], [0.0, 1.02, 1.0]])
    c = (320.0, 180.0)
    M2 = drop_scale(M, c)
    s = math.hypot(M2[0, 0], M2[1, 0])
    assert math.isclose(s, 1.0, abs_tol=1e-9)
    p = np.array([c[0], c[1], 1.0])
    assert np.allclose(M @ p, to3(M2)[:2] @ p)


def test_decompose_identity():
    assert decompose(np.array([[1.0, 0, 0], [0, 1.0, 0]])) == (0.0, 0.0, 0.0, 1.0)


def test_history_composes_inside_window_only():
    h = EgoHistory(max_span_frames=10)
    for f in range(1, 30):
        h.push(f, np.array([[1.0, 0, 2.0], [0, 1.0, 0]]), True)
    M, ok, avail = h.transform_between(20, 25)           # 5 steps of +2 px
    assert avail and ok and math.isclose(M[0, 2], 10.0)
    Minv, _, _ = h.transform_between(25, 20)
    assert math.isclose(Minv[0, 2], -10.0)
    _, _, avail = h.transform_between(1, 28)             # longer than max_span: never composed
    assert not avail


def test_history_flags_invalid_transforms():
    h = EgoHistory(10)
    h.push(1, np.array([[1.0, 0, 0], [0, 1.0, 0]]), True)
    h.push(2, np.array([[1.0, 0, 0], [0, 1.0, 0]]), False)
    _, ok, avail = h.transform_between(0, 2)
    assert not avail or not ok
    _, ok, avail = h.transform_between(1, 2)
    assert avail and not ok
