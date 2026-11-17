import numpy as np

from ptr.schemas import PotholeObs, Track
from ptr.viz.overlay import ActivePair, TrailStore, render


def tr(i, x):
    b = (x, 100.0, x + 30, 160.0)
    return Track(i, 0, 0.0, b, b, 0.9, ((b[0] + b[2]) / 2, b[3]), reliable=(i == 1))


def po(i, confirmed):
    return PotholeObs(i, 0, 0.0, (200.0, 170.0, 250.0, 185.0), 0.8, confirmed, False, 0, 3, "medium")


def test_render_keeps_size_and_does_not_mutate_input(pipe_cfg):
    frame = np.full((360, 640, 3), 90, np.uint8)
    before = frame.copy()
    trails = TrailStore(pipe_cfg.output.trail_length)
    pairs = [ActivePair(1, 1, "High", 0.9, "close pass (distance 0.5 MTW widths), strong lateral shift", (115.0, 160.0),
                        (200.0, 170.0, 250.0, 185.0)),
             ActivePair(2, 2, "Low", 0.1, "far", (315.0, 160.0), (200.0, 170.0, 250.0, 185.0))]
    for k in range(5):
        out = render(frame, [tr(1, 100 + k), tr(2, 300)], [po(1, True), po(2, False)], pairs, trails,
                     pipe_cfg.output, k, k / 30, ego_valid=True)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert np.array_equal(frame, before)
    assert not np.array_equal(out, frame)


def test_render_empty_scene_and_odd_sizes(pipe_cfg):
    for shape in [(360, 640, 3), (100, 120, 3), (1080, 1920, 3)]:
        frame = np.zeros(shape, np.uint8)
        out = render(frame, [], [], [], TrailStore(10), pipe_cfg.output, 0, 0.0, ego_valid=False,
                     camera_mode="static")
        assert out.shape == shape


def test_render_boxes_partly_outside_frame(pipe_cfg):
    frame = np.zeros((100, 100, 3), np.uint8)
    t = Track(1, 0, 0.0, (-20.0, -10.0, 150.0, 130.0), (-20.0, -10.0, 150.0, 130.0), 0.9, (65.0, 130.0))
    out = render(frame, [t], [], [], TrailStore(5), pipe_cfg.output, 0, 0.0)
    assert out.shape == frame.shape
