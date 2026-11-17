import dataclasses

from ptr.analysis.pothole_registry import PotholeRegistry, size_from_area_fraction
from ptr.schemas import Detection


def mk(pipe_cfg, mode="moving", fps=30.0):
    return PotholeRegistry(pipe_cfg.registry, pipe_cfg.features.registry_size, mode, fps, (640, 360))


def det(b, conf=0.8):
    return Detection(b, conf, 0)


def test_stable_id_with_jittering_boxes(pipe_cfg):
    reg = mk(pipe_cfg)
    ids = set()
    for k in range(30):
        j = (k % 3) - 1
        obs = reg.update(k, k / 30, [det((200 + j, 200 + j, 240 + j, 220 + j))])
        ids.update(o.pothole_id for o in obs)
    assert ids == {1}
    assert obs[0].confirmed and not obs[0].last_known


def test_flicker_single_frame_is_never_confirmed(pipe_cfg):
    reg = mk(pipe_cfg)
    seen_confirmed = False
    for k in range(40):
        dets = [det((50, 50, 80, 70))] if k == 5 else []
        for o in reg.update(k, k / 30, dets):
            seen_confirmed |= o.confirmed
    assert not seen_confirmed
    assert reg.n_active == 0                            # tentative entry expired


def test_confirmation_needs_min_hits(pipe_cfg):
    reg = mk(pipe_cfg)
    flags = []
    for k in range(pipe_cfg.registry.min_hits + 1):
        flags.append(reg.update(k, k / 30, [det((100, 100, 140, 120))])[0].confirmed)
    assert flags[:pipe_cfg.registry.min_hits - 1] == [False] * (pipe_cfg.registry.min_hits - 1)
    assert flags[pipe_cfg.registry.min_hits - 1] is True


def test_dropout_keeps_id_and_marks_last_known(pipe_cfg):
    reg = mk(pipe_cfg)
    for k in range(10):
        reg.update(k, k / 30, [det((100, 100, 140, 120))])
    obs = reg.update(10, 10 / 30, [])
    assert len(obs) == 1 and obs[0].last_known and obs[0].last_seen_frame == 9
    obs = reg.update(11, 11 / 30, [det((101, 100, 141, 120))])
    assert obs[0].pothole_id == 1 and not obs[0].last_known


def test_dropped_after_max_age(pipe_cfg):
    reg = mk(pipe_cfg)
    for k in range(10):
        reg.update(k, k / 30, [det((100, 100, 140, 120))])
    for k in range(10, 10 + 40):
        obs = reg.update(k, k / 30, [])
    assert obs == [] and reg.n_dropped == 1


def test_static_mode_is_persistent(pipe_cfg):
    reg = mk(pipe_cfg, mode="static")
    for k in range(10):
        reg.update(k, k / 30, [det((100, 100, 140, 120))])
    for k in range(10, 400):
        obs = reg.update(k, k / 30, [])
    assert len(obs) == 1 and obs[0].last_known


def test_two_potholes_get_two_ids(pipe_cfg):
    reg = mk(pipe_cfg)
    for k in range(6):
        obs = reg.update(k, k / 30, [det((50, 200, 90, 220)), det((400, 220, 450, 245))])
    assert sorted(o.pothole_id for o in obs) == [1, 2]


def test_split_detection_keeps_original_id(pipe_cfg):
    reg = mk(pipe_cfg)
    for k in range(8):
        reg.update(k, k / 30, [det((100, 100, 180, 140))])
    obs = reg.update(8, 8 / 30, [det((100, 100, 140, 140)), det((140, 100, 180, 140))])
    orig = [o for o in obs if o.pothole_id == 1]
    assert len(orig) == 1 and orig[0].confirmed
    assert any(o.pothole_id != 1 and not o.confirmed for o in obs)


def test_merge_of_overlapping_entries(pipe_cfg):
    reg = mk(pipe_cfg)
    for k in range(5):
        reg.update(k, k / 30, [det((100, 100, 140, 120)), det((300, 100, 340, 120))])
    e2 = reg._entries[2]
    e2.boxes.clear()
    e2.boxes.append((e2.last_seen, (101, 100, 141, 120)))
    reg._merge_overlaps()
    assert sorted(reg._entries) == [1] and reg.n_merged == 1


def test_median_box_rejects_outlier(pipe_cfg):
    reg = mk(pipe_cfg)
    boxes = [(100, 100, 140, 120)] * 4 + [(130, 100, 170, 120)]
    for k, b in enumerate(boxes):
        obs = reg.update(k, k / 30, [det(b)])
    assert obs[0].bbox[0] <= 105


def test_size_class_from_area_fraction(pipe_cfg):
    sc = pipe_cfg.features.registry_size
    area = 640 * 360
    assert size_from_area_fraction(0.0005 * area, area, sc) == "small"
    assert size_from_area_fraction(0.005 * area, area, sc) == "medium"
    assert size_from_area_fraction(0.05 * area, area, sc) == "large"
