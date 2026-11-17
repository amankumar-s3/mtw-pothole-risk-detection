"""Persistence + hysteresis and encounter-level events.

  * a HIGHER class is accepted only after it persisted for ``accept_seconds`` (0 = immediately);
  * the class drops only after the lower class persisted for ``deescalate_seconds``;
  * one event per (mtw_id, pothole_id) encounter; it closes after ``event_close_seconds`` without rows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..config import RiskModeCfg
from ..schemas import Event, RISK_LABELS, RiskAssessment, label_rank
from ..logging_setup import get_logger

log = get_logger("events")


@dataclass
class _PairState:
    event_id: str
    accepted: int = 0
    up_count: int = 0
    up_min: int = 99
    down_count: int = 0
    down_max: int = -1
    last_frame: int = -1
    rows: list = field(default_factory=list)   # (assessment, accepted_rank)


class EventAggregator:
    def __init__(self, block: RiskModeCfg, fps: float) -> None:
        p = block.persistence
        self.accept_frames = max(1, math.ceil(p.accept_seconds * fps))
        self.down_frames = max(1, math.ceil(p.deescalate_seconds * fps))
        self.close_frames = max(1, math.ceil(p.event_close_seconds * fps))
        self.min_event_frames = p.min_event_frames
        self._states: dict[tuple[int, int], _PairState] = {}
        self._events: list[Event] = []
        self._n = 0
        self.n_discarded_short = 0

    # ------------------------------------------------------------------ per frame
    def update(self, frame: int, assessments: list[RiskAssessment]) -> dict[tuple[int, int], str]:
        """Feed this frame's assessments; returns the ACCEPTED label per active pair."""
        out: dict[tuple[int, int], str] = {}
        for a in assessments:
            key = (a.mtw_id, a.pothole_id)
            st = self._states.get(key)
            if st is None:
                self._n += 1
                st = self._states[key] = _PairState(event_id=f"E{self._n:04d}")
            self._step(st, label_rank(a.raw_label))
            st.last_frame = frame
            st.rows.append((a, st.accepted))
            out[key] = RISK_LABELS[st.accepted]
        for key, st in list(self._states.items()):
            if frame - st.last_frame > self.close_frames:
                self._close(key)
        return out

    def _step(self, st: _PairState, raw: int) -> None:
        if raw > st.accepted:
            st.up_count += 1
            st.up_min = min(st.up_min, raw)
            st.down_count, st.down_max = 0, -1
            if st.up_count >= self.accept_frames:
                st.accepted = st.up_min
                st.up_count, st.up_min = 0, 99
        elif raw < st.accepted:
            st.down_count += 1
            st.down_max = max(st.down_max, raw)
            st.up_count, st.up_min = 0, 99
            if st.down_count >= self.down_frames:
                st.accepted = st.down_max
                st.down_count, st.down_max = 0, -1
        else:
            st.up_count, st.up_min = 0, 99
            st.down_count, st.down_max = 0, -1

    # ------------------------------------------------------------------ closing
    def _close(self, key: tuple[int, int]) -> None:
        st = self._states.pop(key)
        if len(st.rows) < self.min_event_frames:
            self.n_discarded_short += 1
            return
        peak_rank = max(r for _, r in st.rows)
        cand = [(a, r) for a, r in st.rows if r == peak_rank]
        peak_a = max(cand, key=lambda x: x[0].total)[0]
        first, last = st.rows[0][0], st.rows[-1][0]
        lf = last.features
        encountered = bool(lf is not None and not lf.no_crossing)
        if encountered:
            clearance, from_cross = lf.clearance, True
        else:
            dxs = [a.features.dx for a, _ in st.rows if a.features is not None and math.isfinite(a.features.dx)]
            clearance, from_cross = (min(dxs) if dxs else math.nan), False
        pf = peak_a.features
        self._events.append(Event(
            event_id=st.event_id, mtw_id=key[0], pothole_id=key[1], start_frame=first.frame,
            end_frame=last.frame, start_t=first.t, end_t=last.t, n_frames=len(st.rows),
            peak_class=RISK_LABELS[peak_rank], peak_score=peak_a.total, peak_frame=peak_a.frame,
            peak_t=peak_a.t, reason=f"{RISK_LABELS[peak_rank]}: {peak_a.reason}", encountered=encountered,
            clearance=clearance, clearance_from_crossing=from_cross,
            direction=lf.direction if lf is not None else "", size_class=pf.size_class if pf else "",
            p_score=peak_a.p_score, t_score=peak_a.t_score, s_score=peak_a.s_score, d_score=peak_a.d_score,
            z_score=peak_a.z_score,
        ))

    def finalize(self) -> list[Event]:
        for key in list(self._states):
            self._close(key)
        return sorted(self._events, key=lambda e: (e.start_frame, e.event_id))

    @staticmethod
    def summarize(events: list[Event]) -> dict:
        counts = {lab: 0 for lab in RISK_LABELS}
        per_pothole: dict[str, dict[str, int]] = {}
        for e in events:
            counts[e.peak_class] += 1
            d = per_pothole.setdefault(str(e.pothole_id), {lab: 0 for lab in RISK_LABELS})
            d[e.peak_class] += 1
        return {"n_events": len(events), "events_per_class": counts, "per_pothole_counts": per_pothole}
