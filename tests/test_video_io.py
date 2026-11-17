import cv2
import numpy as np
import pytest

from ptr.config import VideoCfg
from ptr.detection.detector import MockDetections, filter_min_size
from ptr.io.video import VideoWriter, iter_frames, output_size, probe
from ptr.schemas import Detection


def make_video(path, n=20, fps=10.0, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        f = np.full((size[1], size[0], 3), i * 10 % 255, np.uint8)
        w.write(f)
    w.release()


def vcfg(**kw):
    base = dict(fallback_fps=25.0, timestamp_mode="index", frame_stride=1, max_frames=None, resize_width=None,
                writer_fourcc="mp4v")
    base.update(kw)
    return VideoCfg(**base)


def test_probe_and_iterate(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p)
    info = probe(p, 25.0)
    assert info.fps_from_file and abs(info.fps - 10.0) < 0.01 and (info.width, info.height) == (64, 48)
    items = list(iter_frames(p, vcfg()))
    assert len(items) == 20
    assert items[3].index == 3 and abs(items[3].t - 0.3) < 1e-9 and items[3].frame.shape == (48, 64, 3)


def test_stride_and_max_frames(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p)
    items = list(iter_frames(p, vcfg(frame_stride=4)))
    assert [i.video_index for i in items] == [0, 4, 8, 12, 16]
    assert [i.index for i in items] == [0, 1, 2, 3, 4]
    assert abs(items[1].t - 0.4) < 1e-9
    assert len(list(iter_frames(p, vcfg(max_frames=3)))) == 3


def test_resize_scale(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p)
    info = probe(p, 25.0)
    w, h, s = output_size(info, vcfg(resize_width=32))
    assert (w, h, s) == (32, 24, 0.5)
    it = next(iter_frames(p, vcfg(resize_width=32)))
    assert it.frame.shape == (24, 32, 3) and it.scale == 0.5


def test_msec_timestamps_are_monotonic(tmp_path):
    p = tmp_path / "a.mp4"
    make_video(p)
    ts = [i.t for i in iter_frames(p, vcfg(timestamp_mode="msec"))]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4", 25.0)


def test_writer_checks_frame_size(tmp_path):
    with VideoWriter(tmp_path / "o.mp4", 10, (64, 48), "mp4v") as w:
        w.write(np.zeros((48, 64, 3), np.uint8))
        with pytest.raises(ValueError):
            w.write(np.zeros((10, 10, 3), np.uint8))


def test_mock_detections_schema_scale_and_filter():
    doc = {"fps": 30, "frames": [{"frame": 2, "potholes": [{"bbox_xyxy": [10, 10, 40, 30], "conf": 0.7},
                                                           {"bbox_xyxy": [0, 0, 3, 3]}],
                                  "mtws": [{"track_id": 5, "bbox_xyxy": [100, 50, 140, 130], "conf": 0.9}]}]}
    m = MockDetections(doc, min_pothole_px=8, min_mtw_px=16)
    pot, mtw = m.detect(np.zeros((10, 10, 3), np.uint8), 2, 0.5)
    assert len(pot) == 1 and pot[0].bbox == (5.0, 5.0, 20.0, 15.0)
    assert mtw[0].track_id == 5 and mtw[0].cls_id == 3
    assert m.detect(np.zeros((1, 1, 3), np.uint8), 99, 1.0) == ([], [])


def test_mock_detections_rejects_bad_files():
    with pytest.raises(ValueError):
        MockDetections({"nope": []})
    with pytest.raises(ValueError):
        MockDetections({"frames": [{"frame": 0, "mtws": [{"bbox_xyxy": [0, 0, 1, 1]}]}]})


def test_filter_min_size():
    d = [Detection((0, 0, 10, 10), 1, 0), Detection((0, 0, 5, 50), 1, 0)]
    assert len(filter_min_size(d, 8)) == 1
