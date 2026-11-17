"""Ego-motion: frame-to-frame global background motion (Lucas-Kanade + RANSAC partial affine).

Rules (blueprint Rev 2, R4.3 / M6):
  * features are taken OUTSIDE all detected boxes and inside a region of interest;
  * on any failure the transform is the identity and ``valid`` is False (never guessed);
  * transforms are NEVER accumulated over the whole video: :class:`EgoHistory` composes them only
    inside a bounded window (``max_span``).

``M`` maps points of frame t-1 into frame t.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..config import EgoMotionCfg
from ..logging_setup import get_logger
from ..schemas import BBox
from .geometry import map_box

log = get_logger("egomotion")

IDENTITY_2x3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class EgoResult:
    frame: int
    M: np.ndarray                 # 2x3, previous frame -> this frame
    valid: bool
    reason: str                   # "ok" | "disabled" | "first_frame" | failure reason
    n_points: int
    inlier_ratio: float
    tx: float
    ty: float
    rot_deg: float
    scale: float

    def row(self, t: float) -> dict:
        return {"frame": self.frame, "t": t, "valid": int(self.valid), "reason": self.reason,
                "n_points": self.n_points, "inlier_ratio": self.inlier_ratio, "tx": self.tx,
                "ty": self.ty, "rot_deg": self.rot_deg, "scale": self.scale}


def to3(M2: np.ndarray) -> np.ndarray:
    return np.vstack([M2, [0.0, 0.0, 1.0]])


def decompose(M2: np.ndarray) -> tuple[float, float, float, float]:
    """(tx, ty, rotation in degrees, scale) of a partial affine [sR | t]."""
    a, b = M2[0, 0], M2[1, 0]
    scale = math.hypot(a, b)
    rot = math.degrees(math.atan2(b, a))
    return float(M2[0, 2]), float(M2[1, 2]), rot, float(scale)


def drop_scale(M2: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    """Replace [sR | t] by [R | t'] that moves the image centre exactly like the original."""
    s = math.hypot(M2[0, 0], M2[1, 0])
    if s < 1e-9:
        return IDENTITY_2x3.copy()
    R = M2[:, :2] / s
    c = np.array(centre, dtype=float)
    t2 = M2[:, :2] @ c + M2[:, 2] - R @ c
    return np.hstack([R, t2[:, None]])


def identity_result(frame: int, reason: str, valid: bool = False, n: int = 0, ratio: float = math.nan) -> EgoResult:
    return EgoResult(frame, IDENTITY_2x3.copy(), valid, reason, n, ratio, 0.0, 0.0, 0.0, 1.0)


class EgoMotionEstimator:
    def __init__(self, cfg: EgoMotionCfg) -> None:
        self.cfg = cfg

    def _mask(self, shape: tuple[int, int], boxes: Sequence[BBox]) -> np.ndarray:
        h, w = shape
        x0, y0, x1, y1 = self.cfg.roi
        mask = np.zeros((h, w), np.uint8)
        mask[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = 255
        m = self.cfg.box_margin_px
        for b in boxes:
            xa, ya = max(0, int(b[0]) - m), max(0, int(b[1]) - m)
            xb, yb = min(w, int(math.ceil(b[2])) + m), min(h, int(math.ceil(b[3])) + m)
            if xb > xa and yb > ya:
                mask[ya:yb, xa:xb] = 0
        return mask

    def estimate(self, frame: int, prev_gray: np.ndarray, curr_gray: np.ndarray,
                 boxes: Sequence[BBox]) -> EgoResult:
        c = self.cfg
        h, w = prev_gray.shape[:2]
        mask = self._mask((h, w), boxes)
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=c.max_corners, qualityLevel=c.quality_level,
                                      minDistance=c.min_distance, mask=mask, blockSize=c.block_size)
        if pts is None or len(pts) < c.min_points:
            return identity_result(frame, "few_points", n=0 if pts is None else len(pts))
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None,
                                              winSize=(c.lk_win, c.lk_win), maxLevel=c.lk_levels)
        if nxt is None or st is None:
            return identity_result(frame, "flow_failed")
        good = st.ravel() == 1
        p0, p1 = pts[good], nxt[good]
        if len(p0) < c.min_points:
            return identity_result(frame, "few_tracked", n=len(p0))
        M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=c.ransac_thresh_px)
        if M is None or inl is None:
            return identity_result(frame, "ransac_failed", n=len(p0))
        ratio = float(inl.sum()) / float(len(inl))
        if ratio < c.min_inlier_ratio:
            return identity_result(frame, "low_inliers", n=len(p0), ratio=ratio)
        if not c.use_scale:
            M = drop_scale(M, (w / 2.0, h / 2.0))
        tx, ty, rot, sc = decompose(M)
        return EgoResult(frame, M, True, "ok", len(p0), ratio, tx, ty, rot, sc)


class EgoHistory:
    """Bounded store of per-frame transforms; composes them only inside a window."""

    def __init__(self, max_span_frames: int) -> None:
        self.max_span = int(max_span_frames)
        self._d: "OrderedDict[int, tuple[np.ndarray, bool]]" = OrderedDict()

    def push(self, frame: int, M2: np.ndarray, valid: bool) -> None:
        self._d[frame] = (to3(M2), valid)
        while len(self._d) > self.max_span + 2:
            self._d.popitem(last=False)

    def transform_between(self, a: int, b: int) -> tuple[np.ndarray, bool, bool]:
        """3x3 matrix taking coordinates of frame ``a`` to frame ``b``.

        Returns (matrix, all_transforms_valid, available). If the span is not fully available or longer
        than max_span, the identity is returned with available=False.
        """
        if a == b:
            return np.eye(3), True, True
        if abs(b - a) > self.max_span:
            return np.eye(3), False, False
        lo, hi = (a, b) if a < b else (b, a)
        M = np.eye(3)
        ok = True
        for f in range(lo + 1, hi + 1):
            if f not in self._d:
                return np.eye(3), False, False
            m, v = self._d[f]
            M = m @ M
            ok = ok and v
        if a > b:
            M = np.linalg.inv(M)
        return M, ok, True

    def map_box(self, box: BBox, a: int, b: int) -> tuple[BBox, bool]:
        """Map ``box`` from frame ``a`` to frame ``b``. Returns (box, ego_ok)."""
        M, valid, avail = self.transform_between(a, b)
        return map_box(M, box), (valid and avail)
