import math

from ptr.schemas import Detection
from ptr.tracking.trajectory import TrajectoryStore


def feed(store, n, fps, box_fn, tid=1):
    out = []
    for k in range(n):
        b = box_fn(k)
        out += store.update(k, k / fps, [Detection(b, 0.9, 3, tid)])
    return out


def test_smoothing_reduces_jitter(pipe_cfg):
    store = TrajectoryStore(pipe_cfg.trajectory)
    jitter = [0, 4, -4, 4, -4, 4, -4, 4, -4, 4]
    tr = feed(store, 10, 30, lambda k: (100 + jitter[k], 100, 120 + jitter[k], 140))
    raw_range = max(t.bbox[0] for t in tr[3:]) - min(t.bbox[0] for t in tr[3:])
    sm_range = max(t.sbbox[0] for t in tr[3:]) - min(t.sbbox[0] for t in tr[3:])
    assert sm_range < raw_range


def test_straight_path_velocity_and_heading(pipe_cfg):
    store = TrajectoryStore(pipe_cfg.trajectory)
    # moves 3 px / frame to the right at 30 fps => 90 px / s, heading 0 degrees
    tr = feed(store, 20, 30, lambda k: (100 + 3 * k, 100, 120 + 3 * k, 140))
    last = tr[-1]
    assert math.isclose(last.vx, 90.0, rel_tol=0.1)
    assert abs(last.vy) < 1.0
    assert abs(last.heading_deg) < 2.0


def test_curved_path_heading_turns(pipe_cfg):
    store = TrajectoryStore(pipe_cfg.trajectory)
    tr = feed(store, 40, 30, lambda k: (100 + 3 * k, 100 + (0 if k < 20 else 3 * (k - 20)),
                                         120 + 3 * k, 140 + (0 if k < 20 else 3 * (k - 20))))
    assert abs(tr[15].heading_deg) < 5
    assert 30 < tr[-1].heading_deg < 60            # image y points down => positive angle


def test_min_history_guard(pipe_cfg):
    store = TrajectoryStore(pipe_cfg.trajectory)
    tr = feed(store, 10, 30, lambda k: (100 + k, 100, 120 + k, 140))
    k0 = pipe_cfg.trajectory.min_history_frames
    assert all(math.isnan(t.speed) for t in tr[:k0 - 1])
    assert math.isfinite(tr[-1].speed)


def test_standstill_heading_nan(pipe_cfg):
    store = TrajectoryStore(pipe_cfg.trajectory)
    tr = feed(store, 12, 30, lambda k: (100, 100, 120, 140))
    assert math.isnan(tr[-1].heading_deg)
    assert tr[-1].speed == 0.0


def test_pruning_and_reliability(pipe_cfg):
    store = TrajectoryStore(pipe_cfg.trajectory)
    feed(store, 10, 30, lambda k: (100 + k, 100, 120 + k, 140), tid=7)
    assert 7 in store.active_ids()
    far = pipe_cfg.trajectory.max_missed_frames + 12
    store.update(far, far / 30, [])
    assert 7 not in store.active_ids()
    assert 7 in store.unreliable_ids()               # lived ~0.3 s < min_reliable_seconds


def test_reliable_flag_after_long_track(pipe_cfg):
    store = TrajectoryStore(pipe_cfg.trajectory)
    tr = feed(store, 45, 30, lambda k: (100 + k, 100, 120 + k, 140))
    assert not tr[5].reliable and tr[-1].reliable
