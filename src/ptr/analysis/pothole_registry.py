"""Short-term pothole stabiliser: IoU + constant-velocity association, min-hits confirmation, max-age drop.

Moving camera (default): IDs are short-term smoothing only, NOT a map. Static camera: confirmed potholes
can be kept for the whole video (``registry.static_persistent``).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

from ..config import RegistryCfg, RegistrySizeCfg
from ..logging_setup import get_logger
from ..schemas import BBox, Detection, PotholeObs
from .egomotion import EgoHistory
from .geometry import box_area, iou

log = get_logger("registry")


@dataclass
class _Entry:
    pid: int
    born: int
    last_seen: int
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    conf: float = 0.0
    last_box: BBox = (0, 0, 0, 0)
    boxes: deque = field(default_factory=lambda: deque(maxlen=5))   # (frame, box in that frame's coordinates)
    vel: tuple[float, float] = (0.0, 0.0)  # residual centre velocity (after ego compensation), px / frame

    def median_box(self, ref: int, ego: Optional[EgoHistory] = None) -> BBox:
        """Per-coordinate median of the stored boxes, all expressed in frame ``ref`` coordinates."""
        mapped = [(ego.map_box(b, f, ref)[0] if ego is not None else b) for f, b in self.boxes]
        m = np.median(np.asarray(mapped, dtype=float), axis=0)
        return (float(m[0]), float(m[1]), float(m[2]), float(m[3]))

    def predicted_box(self, frame: int, ego: Optional[EgoHistory] = None) -> BBox:
        """Where the pothole should be in ``frame``: last median + residual velocity, mapped by ego-motion."""
        b = self.median_box(self.last_seen, ego)
        dt = frame - self.last_seen
        dx, dy = self.vel[0] * dt, self.vel[1] * dt
        pred = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
        return ego.map_box(pred, self.last_seen, frame)[0] if ego is not None else pred


def size_from_area_fraction(area: float, frame_area: float, cfg: RegistrySizeCfg) -> str:
    frac = area / frame_area if frame_area > 0 else 0.0
    if frac < cfg.frac_small_max:
        return "small"
    if frac >= cfg.frac_large_min:
        return "large"
    return "medium"


class PotholeRegistry:
    def __init__(self, cfg: RegistryCfg, size_cfg: RegistrySizeCfg, camera_mode: str, fps: float,
                 frame_size: tuple[int, int]) -> None:
        self.cfg = cfg
        self.size_cfg = size_cfg
        self.camera_mode = camera_mode
        self.frame_area = float(frame_size[0] * frame_size[1])
        self.max_age = max(1, math.ceil(cfg.max_age_seconds * fps))
        self.tent_age = max(1, math.ceil(cfg.tentative_max_age_seconds * fps))
        self._entries: dict[int, _Entry] = {}
        self._next = 1
        self.n_dropped = 0
        self.n_merged = 0

    # ------------------------------------------------------------------ main step
    def update(self, frame: int, t: float, dets: list[Detection],
               ego: Optional[EgoHistory] = None) -> list[PotholeObs]:
        """``ego`` (optional) lets association and median filtering work in a shake-compensated frame."""
        entries = list(self._entries.values())
        pairs = []
        for di, d in enumerate(dets):
            for ei, e in enumerate(entries):
                v = iou(d.bbox, e.predicted_box(frame, ego))
                if v >= self.cfg.iou_match:
                    pairs.append((v, di, ei))
        pairs.sort(reverse=True)
        used_d: set[int] = set()
        used_e: set[int] = set()
        for v, di, ei in pairs:
            if di in used_d or ei in used_e:
                continue
            used_d.add(di)
            used_e.add(ei)
            self._match(entries[ei], dets[di], frame, ego)
        for di, d in enumerate(dets):
            if di not in used_d:
                self._spawn(d, frame)
        for ei, e in enumerate(entries):
            if ei not in used_e:
                e.missed += 1
        self._drop_expired()
        self._merge_overlaps(frame, ego)
        return [self._obs(e, frame, t, ego) for e in sorted(self._entries.values(), key=lambda x: x.pid)]

    # ------------------------------------------------------------------ internals
    def _match(self, e: _Entry, d: Detection, frame: int, ego: Optional[EgoHistory]) -> None:
        old_box = ego.map_box(e.last_box, e.last_seen, frame)[0] if ego is not None else e.last_box
        old_c = ((old_box[0] + old_box[2]) / 2, (old_box[1] + old_box[3]) / 2)
        new_c = ((d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2)
        dt = max(1, frame - e.last_seen)
        v_new = ((new_c[0] - old_c[0]) / dt, (new_c[1] - old_c[1]) / dt)
        a = self.cfg.velocity_alpha
        e.vel = (a * v_new[0] + (1 - a) * e.vel[0], a * v_new[1] + (1 - a) * e.vel[1])
        e.boxes.append((frame, d.bbox))
        e.last_box = d.bbox
        e.last_seen = frame
        e.conf = d.conf
        e.hits += 1
        e.missed = 0
        if e.hits >= self.cfg.min_hits:
            e.confirmed = True

    def _spawn(self, d: Detection, frame: int) -> None:
        e = _Entry(pid=self._next, born=frame, last_seen=frame, conf=d.conf, last_box=d.bbox,
                   boxes=deque([(frame, d.bbox)], maxlen=self.cfg.median_window))
        e.confirmed = e.hits >= self.cfg.min_hits
        self._entries[e.pid] = e
        self._next += 1

    def _drop_expired(self) -> None:
        for pid, e in list(self._entries.items()):
            if e.confirmed:
                if self.camera_mode == "static" and self.cfg.static_persistent:
                    continue
                limit = self.max_age
            else:
                limit = self.tent_age
            if e.missed > limit:
                del self._entries[pid]
                self.n_dropped += 1

    def _merge_overlaps(self, frame: int = 0, ego: Optional[EgoHistory] = None) -> None:
        ids = sorted(self._entries)
        gone: set[int] = set()
        for i, a in enumerate(ids):
            if a in gone:
                continue
            for b in ids[i + 1:]:
                if b in gone:
                    continue
                ea, eb = self._entries[a], self._entries[b]
                if iou(ea.median_box(frame, ego), eb.median_box(frame, ego)) > self.cfg.merge_iou:
                    keep, drop = self._entries[a], self._entries[b]  # older id wins
                    keep.hits = max(keep.hits, drop.hits)
                    keep.confirmed = keep.confirmed or drop.confirmed
                    keep.missed = min(keep.missed, drop.missed)
                    gone.add(b)
                    self.n_merged += 1
        for b in gone:
            del self._entries[b]

    def _obs(self, e: _Entry, frame: int, t: float, ego: Optional[EgoHistory]) -> PotholeObs:
        # bbox is expressed in the coordinates of frame ``last_seen`` (== ``frame`` while the pothole is matched)
        box = e.median_box(e.last_seen, ego)
        return PotholeObs(
            pothole_id=e.pid, frame=frame, t=t, bbox=box, conf=e.conf, confirmed=e.confirmed,
            last_known=e.missed > 0, last_seen_frame=e.last_seen, hits=e.hits,
            size_class=size_from_area_fraction(box_area(box), self.frame_area, self.size_cfg),
        )

    @property
    def n_active(self) -> int:
        return len(self._entries)
