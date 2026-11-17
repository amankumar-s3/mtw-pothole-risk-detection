import math

import numpy as np

from ptr.analysis.geometry import (contact_point, depth_gap, iou, lateral_gap, lateral_offset, map_box)


def test_contact_point_is_bottom_centre():
    assert contact_point((10, 20, 30, 60)) == (20.0, 60)


def test_depth_gap_sign_convention():
    poth = (100, 100, 120, 110)                      # bottom row 110
    closer = (100, 100, 120, 150)                    # MTW contact row 150 > 110 => closer to camera
    farther = (100, 60, 120, 90)
    assert depth_gap(closer, poth) > 0
    assert depth_gap(farther, poth) < 0
    assert math.isclose(depth_gap(closer, poth), (150 - 110) / 50)
    assert depth_gap((0, 0, 10, 110), poth) == 0.0


def test_lateral_gap_inside_and_outside():
    poth = (100, 100, 140, 110)
    mtw_in = (110, 50, 130, 120)                     # centre 120 inside
    mtw_out = (170, 50, 190, 120)                    # centre 180: 40 px right of the pothole's right edge
    assert lateral_gap(mtw_in, poth) == 0.0
    assert math.isclose(lateral_gap(mtw_out, poth), 40 / 20)
    mtw_left = (30, 50, 50, 120)                     # centre 40: 60 px left of left edge
    assert math.isclose(lateral_gap(mtw_left, poth), 60 / 20)


def test_lateral_offset_sign_and_mirror():
    poth = (100, 100, 140, 110)
    right = (130, 50, 150, 120)
    left = (90, 50, 110, 120)
    assert lateral_offset(right, poth) > 0 > lateral_offset(left, poth)
    assert math.isclose(lateral_offset(right, poth), -lateral_offset(left, poth))


def test_degenerate_boxes_give_nan():
    assert math.isnan(depth_gap((0, 5, 10, 5), (0, 0, 1, 1)))
    assert math.isnan(lateral_gap((5, 0, 5, 10), (0, 0, 1, 1)))


def test_iou_known_values():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert math.isclose(iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)
    assert iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0.0


def test_map_box_translation():
    M = np.array([[1, 0, 5], [0, 1, -3], [0, 0, 1.0]])
    assert map_box(M, (0, 0, 10, 10)) == (5, -3, 15, 7)


def test_perspective_correction_changes_offset_only_when_depths_differ():
    mtw_far = (300, 150, 310, 175)                   # contact row 175, horizon 150 -> 25 below
    poth_near = (400, 190, 430, 200)                 # bottom row 200 -> 50 below horizon (closer)
    raw = lateral_offset(mtw_far, poth_near)
    corr = lateral_offset(mtw_far, poth_near, persp=(320.0, 150.0, 3.0))
    assert not math.isclose(raw, corr)
    same_depth = (400, 150, 410, 200)
    assert math.isclose(lateral_offset(same_depth, poth_near, persp=(320.0, 150.0, 3.0)),
                        lateral_offset(same_depth, poth_near))


def test_perspective_guard_falls_back_for_implausible_horizon():
    mtw = (300, 150, 310, 160)                       # contact row 160: only 10 px below a horizon at 150
    poth = (400, 150, 430, 230)                      # bottom row 230: 80 px below => ratio 0.125 < 1/3
    assert lateral_offset(mtw, poth, persp=(320.0, 150.0, 3.0)) == lateral_offset(mtw, poth)
