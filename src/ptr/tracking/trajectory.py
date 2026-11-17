"""Per-track history: EMA smoothing of the box, contact point, velocity and heading."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ..config import TrajectoryCfg
from ..schemas import BBox, Detection, Track
from ..analysis.geometry import contact_point
from ..logging_setup import get_logger

log = get_logger("trajectory")


@dataclass
class _State:
    track_id: int
    first_t: float
    first_frame: int
    last_frame: int
    sbbox: BBox
    hist: deque = field(default_factory=lambda: deque(maxlen=64))  # (frame, t, contact_x, contact_y)
    n: int = 0


def ema_box(prev: BBox, new: BBox, alpha: float) -> BBox:
    return tuple(alpha * n + (1.0 - alpha) * p for p, n in zip(prev, new))  # type: ignore[return-value]


class TrajectoryStore:
    """Holds one smoothed trajectory per MTW id; emits :class:`Track` records per frame."""

    def __init__(self, cfg: TrajectoryCfg) -> None:
        self.cfg = cfg
        self._states: dict[int, _State] = {}
        self._lifetimes: dict[int, float] = {}

    def update(self, frame: int, t: float, dets: list[Detection]) -> list[Track]:
        out: list[Track] = []
        for d in dets:
            if d.track_id is None:
                continue
            st = self._states.get(d.track_id)
            if st is None:
                st = _State(d.track_id, t, frame, frame, d.bbox)
                self._states[d.track_id] = st
                sb = d.bbox
            else:
                sb = ema_box(st.sbbox, d.bbox, self.cfg.ema_alpha)
            st.sbbox = sb
            st.last_frame = frame
            st.n += 1
            cx, cy = contact_point(sb)
            st.hist.append((frame, t, cx, cy))
            vx = vy = speed = heading = math.nan
            if st.n >= self.cfg.min_history_frames and len(st.hist) > 1:
                k = min(self.cfg.velocity_frames, len(st.hist) - 1)
                f0, t0, x0, y0 = st.hist[-1 - k]
                dt = t - t0
                if dt > 0:
                    vx, vy = (cx - x0) / dt, (cy - y0) / dt
                    speed = math.hypot(vx, vy)
                    heading = math.degrees(math.atan2(vy, vx)) if speed > 1e-9 else math.nan
            out.append(Track(
                track_id=d.track_id, frame=frame, t=t, bbox=d.bbox, sbbox=sb, conf=d.conf, contact=(cx, cy),
                vx=vx, vy=vy, speed=speed, heading_deg=heading, age_frames=st.n,
                reliable=(t - st.first_t) >= self.cfg.min_reliable_seconds,
            ))
        self._prune(frame)
        return out

    def _prune(self, frame: int) -> None:
        dead = [i for i, s in self._states.items() if frame - s.last_frame > self.cfg.max_missed_frames]
        for i in dead:
            s = self._states.pop(i)
            self._lifetimes[i] = (s.hist[-1][1] - s.first_t) if s.hist else 0.0

    def finalize(self) -> dict[int, float]:
        """Lifetimes (seconds) of all tracks seen; tracks shorter than min_reliable_seconds are unreliable."""
        for i, s in list(self._states.items()):
            self._lifetimes[i] = (s.hist[-1][1] - s.first_t) if s.hist else 0.0
        self._states.clear()
        return dict(self._lifetimes)

    def unreliable_ids(self) -> list[int]:
        return sorted(i for i, life in self.finalize().items() if life < self.cfg.min_reliable_seconds)

    def active_ids(self) -> list[int]:
        return sorted(self._states)

    def history(self, track_id: int) -> Optional[list[tuple[int, float, float, float]]]:
        st = self._states.get(track_id)
        return list(st.hist) if st else None
