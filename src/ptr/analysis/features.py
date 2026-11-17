"""MTW-pothole interaction features (moving-camera ground-plane proxy; see analysis/geometry.py).

Per (mtw_id, pothole_id) pair and frame the extractor emits one :class:`InteractionFeatures` row while the
pair is "in range" (|dy| <= max_abs_dy and dx <= max_dx). Everything is computed over a sliding window of
``features.window_seconds`` after bringing all samples into the CURRENT frame's coordinates with the
ego-motion transforms of that window only (never accumulated over the video).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import FeaturesCfg
from ..logging_setup import get_logger
from ..schemas import BBox, InteractionFeatures, PotholeObs, Track
from .egomotion import EgoHistory
from .geometry import box_area, box_h, box_w, contact_point, depth_gap, lateral_gap, lateral_offset

log = get_logger("features")
NAN = math.nan


@dataclass
class _Sample:
    frame: int
    t: float
    mtw_box: BBox
    poth_box: BBox
    poth_src: int          # frame in whose coordinates poth_box is expressed
    last_known: bool


@dataclass
class _Pair:
    mtw_id: int
    pothole_id: int
    samples: deque = field(default_factory=deque)
    last_frame: int = -1
    poth_box: Optional[BBox] = None       # last pothole box seen (for memory)
    poth_seen: int = -1
    prev_dy: float = NAN
    prev_dx: float = NAN
    encountered: bool = False
    enc_frame: int = -1
    clearance: float = NAN
    direction: str = ""


def _slope(ts: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares slope dy/dt; NaN if undefined."""
    if len(ts) < 2 or np.ptp(ts) <= 0:
        return NAN
    return float(np.polyfit(ts, ys, 1)[0])


class FeatureExtractor:
    def __init__(self, cfg: FeaturesCfg, fps: float, frame_size: tuple[int, int], min_track_age: int,
                 ego: Optional[EgoHistory] = None) -> None:
        self.cfg = cfg
        self.fps = fps
        self.frame_area = float(frame_size[0] * frame_size[1])
        self.min_track_age = min_track_age
        self.ego = ego if cfg.ego_compensation else None
        self.memory_frames = max(1, math.ceil(cfg.pothole_memory_seconds * fps))
        self.expire_frames = max(1, math.ceil(cfg.pair_expire_seconds * fps))
        self.post_frames = math.ceil(cfg.post_encounter_seconds * fps)
        pc = cfg.perspective
        self.persp = (pc.principal_x_frac * frame_size[0], pc.horizon_row_frac * frame_size[1],
                      pc.max_depth_ratio) if pc.enabled else None
        self._pairs: dict[tuple[int, int], _Pair] = {}
        self._done: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------ helpers
    def _map(self, box: BBox, a: int, b: int) -> tuple[BBox, bool]:
        if self.ego is None or a == b:
            return box, True
        return self.ego.map_box(box, a, b)

    def _in_range(self, mtw: BBox, poth: BBox) -> tuple[bool, float, float]:
        dy = depth_gap(mtw, poth)
        dx = lateral_gap(mtw, poth)
        ok = math.isfinite(dy) and math.isfinite(dx) and abs(dy) <= self.cfg.max_abs_dy and dx <= self.cfg.max_dx
        return ok, dy, dx

    # ------------------------------------------------------------------ main step
    def update(self, frame: int, t: float, tracks: list[Track], potholes: list[PotholeObs]) -> list[InteractionFeatures]:
        mtws = {tr.track_id: tr for tr in tracks if tr.age_frames >= self.min_track_age}
        confirmed = {p.pothole_id: p for p in potholes if p.confirmed}
        rows: list[InteractionFeatures] = []
        handled: set[tuple[int, int]] = set()

        for mid, tr in mtws.items():
            for pid, p in confirmed.items():
                key = (mid, pid)
                if key in self._done:
                    continue
                pb, ego_ok = self._map(p.bbox, p.last_seen_frame, frame) if p.last_known else (p.bbox, True)
                in_range, _, _ = self._in_range(tr.sbbox, pb)
                pair = self._pairs.get(key)
                if pair is None and not in_range:
                    continue
                if pair is None:
                    pair = self._pairs[key] = _Pair(mid, pid)
                if not p.last_known:
                    pair.poth_box, pair.poth_seen = p.bbox, frame
                handled.add(key)
                if not in_range:
                    continue
                src = p.last_seen_frame if p.last_known else frame
                raw_pb = p.bbox
                row = self._add_and_compute(pair, frame, t, tr.sbbox, raw_pb, src, p.last_known)
                if row is not None:
                    rows.append(row)

        # pothole no longer reported by the registry: use last-known box from pair memory
        for key, pair in list(self._pairs.items()):
            if key in handled or key in self._done:
                continue
            mid, pid = key
            if mid in mtws and pid not in confirmed and pair.poth_box is not None \
                    and frame - pair.poth_seen <= self.memory_frames:
                pb, _ = self._map(pair.poth_box, pair.poth_seen, frame)
                in_range, _, _ = self._in_range(mtws[mid].sbbox, pb)
                if in_range:
                    row = self._add_and_compute(pair, frame, t, mtws[mid].sbbox, pair.poth_box, pair.poth_seen, True)
                    if row is not None:
                        rows.append(row)
                    handled.add(key)

        for key, pair in list(self._pairs.items()):
            if key in self._done:
                continue
            if pair.encountered and self.post_frames >= 0 and frame - pair.enc_frame >= self.post_frames:
                self._done.add(key)
                del self._pairs[key]
            elif frame - pair.last_frame > self.expire_frames:
                del self._pairs[key]
        return rows

    # ------------------------------------------------------------------ per-pair computation
    def _add_and_compute(self, pair: _Pair, frame: int, t: float, mtw_box: BBox, poth_box: BBox,
                         poth_src: int, last_known: bool) -> Optional[InteractionFeatures]:
        cfg = self.cfg
        pair.samples.append(_Sample(frame, t, mtw_box, poth_box, poth_src, last_known))
        pair.last_frame = frame
        while pair.samples and t - pair.samples[0].t > cfg.window_seconds + 1e-9:
            pair.samples.popleft()

        # bring all window samples to the current frame
        ts, dys, dxs, offs, cys, hs = [], [], [], [], [], []
        ego_ok_all = True
        for s in pair.samples:
            mb, ok1 = self._map(s.mtw_box, s.frame, frame)
            pb, ok2 = self._map(s.poth_box, s.poth_src, frame)
            ego_ok_all = ego_ok_all and ok1 and ok2
            dy, dx = depth_gap(mb, pb), lateral_gap(mb, pb)
            off = lateral_offset(mb, pb, self.persp)
            if not (math.isfinite(dy) and math.isfinite(dx) and math.isfinite(off)):
                continue
            ts.append(s.t)
            dys.append(dy)
            dxs.append(dx)
            offs.append(off)
            cys.append(contact_point(mb)[1])
            hs.append(box_h(mb))
        if not ts:
            return None
        ts_a, dys_a, dxs_a, offs_a = np.array(ts), np.array(dys), np.array(dxs), np.array(offs)
        dy_now, dx_now = dys[-1], dxs[-1]
        dist = np.hypot(dxs_a, dys_a)
        n = len(ts)
        insufficient = n < cfg.min_samples

        # encounter = zero crossing of dy (sign convention: analysis/geometry.py)
        encounter = False
        if not pair.encountered and math.isfinite(pair.prev_dy):
            pd, nd = pair.prev_dy, dy_now
            if (pd > 0 and nd <= 0) or (pd < 0 and nd >= 0):
                encounter = True
                pair.encountered = True
                pair.enc_frame = frame
                pair.direction = "away" if pd > 0 else "oncoming"
                den = abs(pd) + abs(nd)
                frac = abs(pd) / den if den > 0 else 0.5
                pair.clearance = pair.prev_dx + (dx_now - pair.prev_dx) * frac if math.isfinite(pair.prev_dx) else dx_now
        pair.prev_dy, pair.prev_dx = dy_now, dx_now

        # time-to-reach: |dy| / (-d|dy|/dt) only while |dy| is shrinking
        ttr = NAN
        if not insufficient:
            sl = _slope(ts_a, np.abs(dys_a))
            if math.isfinite(sl) and sl < -cfg.min_slope_eps and abs(dy_now) > 0:
                ttr = abs(dy_now) / (-sl)

        # swerve = slope of the lateral offset (MTW widths / s)
        swerve_signed = NAN
        if not insufficient:
            swerve_signed = _slope(ts_a, offs_a)
        swerve = abs(swerve_signed) if math.isfinite(swerve_signed) else NAN

        # deceleration: relative drop of the depth-rate proxy q = |d contact_row / dt| / box_height^2
        decel, standstill = self._decel(ts_a, np.array(cys), np.array(hs), insufficient)

        # size class
        size_ratio, size_class, size_source = self._size(box_w, pair, frame, dy_now)

        if pair.encountered:
            clearance, from_cross = pair.clearance, True
        else:
            clearance, from_cross = float(np.min(dxs_a)), False

        return InteractionFeatures(
            frame=frame, t=t, mtw_id=pair.mtw_id, pothole_id=pair.pothole_id, dy=dy_now, dx=dx_now,
            dist=float(dist[-1]), proximity=float(np.min(dist)), ttr=ttr, swerve=swerve,
            swerve_signed=swerve_signed, decel=decel, clearance=clearance, clearance_from_crossing=from_cross,
            encounter=encounter, encounter_frame=pair.enc_frame, direction=pair.direction,
            size_ratio=size_ratio, size_class=size_class, size_source=size_source, n_samples=n,
            insufficient_history=insufficient, standstill=standstill,
            ego_invalid=(self.ego is not None and not ego_ok_all), pothole_last_known=last_known,
            no_crossing=not pair.encountered,
        )

    def _decel(self, ts: np.ndarray, cys: np.ndarray, hs: np.ndarray, insufficient: bool) -> tuple[float, bool]:
        if insufficient or len(ts) < 4:
            return NAN, False
        dts = np.diff(ts)
        ok = dts > 0
        if ok.sum() < 3:
            return NAN, False
        vy = np.diff(cys)[ok] / dts[ok]
        hmid = 0.5 * (hs[1:] + hs[:-1])[ok]
        if np.any(hmid <= 0):
            return NAN, False
        speed_h = np.abs(np.diff(cys)[ok]) / hmid / dts[ok]          # MTW heights / s (standstill test)
        if float(np.mean(speed_h)) < self.cfg.standstill_speed:
            return NAN, True
        q = np.abs(vy) / hmid ** 2
        half = len(q) // 2
        q1, q2 = float(np.mean(q[:half])), float(np.mean(q[half:]))
        if q1 <= 0:
            return NAN, False
        return (q1 - q2) / q1, False

    def _size(self, _unused, pair: _Pair, frame: int, dy_now: float) -> tuple[float, str, str]:
        s = pair.samples[-1]
        mb, _ = self._map(s.mtw_box, s.frame, frame)
        pb, _ = self._map(s.poth_box, s.poth_src, frame)
        sc = self.cfg.size
        if abs(dy_now) <= sc.max_abs_dy and box_w(mb) > 0:
            r = box_w(pb) / box_w(mb)
            cls = "small" if r < sc.ratio_small_max else ("large" if r >= sc.ratio_large_min else "medium")
            return r, cls, "ratio"
        frac = box_area(pb) / self.frame_area if self.frame_area > 0 else 0.0
        cls = "small" if frac < sc.frac_small_max else ("large" if frac >= sc.frac_large_min else "medium")
        return NAN, cls, "area_fraction"
