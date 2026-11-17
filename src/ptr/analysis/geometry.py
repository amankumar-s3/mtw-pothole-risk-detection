"""Ground-plane proxy geometry. The sign convention is defined ONCE, here, and tested in tests/.

Image coordinates: x to the right, y DOWN. On a forward-facing camera over a roughly flat road,
the image row of a ground contact point INCREASES as the object gets closer to the camera.

    contact point of an MTW   = bottom-centre of its box:  (cx, y2)
    pothole "bottom row"      = y2 of the pothole box (its nearest edge to the camera)

    dy = (mtw_contact_row - pothole_bottom_row) / mtw_box_height
         dy > 0 : MTW is closer to the camera than the pothole
         dy < 0 : MTW is farther than the pothole
         An ENCOUNTER is a zero crossing of dy:
           "away"     : dy goes + -> -  (rider moving away from the camera vehicle, passing the pothole)
           "oncoming" : dy goes - -> +  (rider coming toward the camera vehicle, passing the pothole)

    dx = horizontal distance between the MTW contact point and the pothole box
         (0 if the contact x lies inside the box's x-range) / mtw_box_width            (always >= 0)

    lateral offset = (mtw_contact_x - pothole_centre_x) / mtw_box_width                (signed, used for swerve)
"""
from __future__ import annotations

import math

import numpy as np

from ..schemas import BBox

EPS = 1e-9


def box_w(b: BBox) -> float:
    return max(0.0, b[2] - b[0])


def box_h(b: BBox) -> float:
    return max(0.0, b[3] - b[1])


def box_area(b: BBox) -> float:
    return box_w(b) * box_h(b)


def box_cx(b: BBox) -> float:
    return 0.5 * (b[0] + b[2])


def contact_point(b: BBox) -> tuple[float, float]:
    """Bottom-centre of a box: the best image proxy of the wheel-ground contact."""
    return (box_cx(b), b[3])


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > EPS else 0.0


def depth_gap(mtw: BBox, poth: BBox) -> float:
    """dy as documented in the module docstring. NaN if the MTW box has no height."""
    h = box_h(mtw)
    if h < EPS:
        return math.nan
    return (mtw[3] - poth[3]) / h


def lateral_gap(mtw: BBox, poth: BBox) -> float:
    """dx >= 0: 0 if the MTW contact x is inside the pothole box, else the distance to the nearest edge."""
    w = box_w(mtw)
    if w < EPS:
        return math.nan
    cx = box_cx(mtw)
    if poth[0] <= cx <= poth[2]:
        return 0.0
    return min(abs(cx - poth[0]), abs(cx - poth[2])) / w


def lateral_offset(mtw: BBox, poth: BBox, persp: tuple[float, float, float] | None = None) -> float:
    """Signed (mtw contact x - pothole centre x) / mtw width (positive = MTW to the right of the pothole).

    ``persp=(principal_x, horizon_row, max_depth_ratio)`` enables the OPTIONAL depth matching: on a flat road the horizontal
    offset of a ground point from the principal point scales with 1 / depth, and 1 / depth is proportional to
    (row - horizon_row), so the pothole centre is rescaled to the MTW's depth before the subtraction.
    Without it, an off-axis pothole produces a spurious lateral drift while the MTW is at another depth.
    """
    w = box_w(mtw)
    if w < EPS:
        return math.nan
    pcx = box_cx(poth)
    if persp is not None:
        cx0, hrow, rmax = persp
        vm, vp = mtw[3] - hrow, poth[3] - hrow
        if vm > EPS and vp > EPS and 1.0 / rmax <= vm / vp <= rmax:
            pcx = cx0 + (pcx - cx0) * (vm / vp)   # u - cx ~ X / Z and Zp / Zm = vm / vp
    return (box_cx(mtw) - pcx) / w


def map_point(M3: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = M3 @ np.array([x, y, 1.0])
    return float(v[0]), float(v[1])


def map_box(M3: np.ndarray, b: BBox) -> BBox:
    """Axis-aligned bounding box of the four mapped corners (M3 is a 3x3 homogeneous affine)."""
    pts = np.array([[b[0], b[1], 1], [b[2], b[1], 1], [b[2], b[3], 1], [b[0], b[3], 1]], dtype=float)
    out = (M3 @ pts.T).T
    return (float(out[:, 0].min()), float(out[:, 1].min()), float(out[:, 0].max()), float(out[:, 1].max()))


def clip_box(b: BBox, w: int, h: int) -> BBox:
    return (min(max(b[0], 0.0), w), min(max(b[1], 0.0), h), min(max(b[2], 0.0), w), min(max(b[3], 0.0), h))
