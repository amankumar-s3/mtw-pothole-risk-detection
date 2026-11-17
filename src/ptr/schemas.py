"""Typed records exchanged between modules (no raw arrays cross module boundaries)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

RISK_LABELS = ("Low", "Moderate", "High")
SIZE_LABELS = ("small", "medium", "large")


def label_rank(label: str) -> int:
    """Ordinal rank of a risk label (Low=0, Moderate=1, High=2)."""
    return RISK_LABELS.index(label)


@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls_id: int
    track_id: Optional[int] = None  # set only for tracked MTWs


@dataclass
class Track:
    """One MTW in one frame. ``bbox`` is the raw tracker box, ``sbbox`` the EMA-smoothed box."""
    track_id: int
    frame: int
    t: float
    bbox: BBox
    sbbox: BBox
    conf: float
    contact: tuple[float, float]  # smoothed bottom-centre
    vx: float = math.nan          # px / s (smoothed contact point)
    vy: float = math.nan
    speed: float = math.nan
    heading_deg: float = math.nan
    age_frames: int = 1
    reliable: bool = False
    missed: int = 0               # >0 means the box is a coasting (not detected this frame) one


@dataclass
class PotholeObs:
    pothole_id: int
    frame: int
    t: float
    bbox: BBox                    # running-median box (last matched box when last_known)
    conf: float
    confirmed: bool
    last_known: bool              # True when not detected in this frame
    last_seen_frame: int
    hits: int
    size_class: str


@dataclass
class InteractionFeatures:
    """Features of one (MTW, pothole) pair at one frame. See docs/methodology.md for the convention.

    dy = (mtw_contact_row - pothole_bottom_row) / mtw_box_height   (larger row = closer to camera)
    dx = horizontal gap between MTW contact point and pothole box (0 if inside) / mtw_box_width
    """
    frame: int
    t: float
    mtw_id: int
    pothole_id: int
    dy: float
    dx: float
    dist: float                   # sqrt(dx^2 + dy^2) at this frame
    proximity: float              # window minimum of dist
    ttr: float                    # seconds, NaN if not closing
    swerve: float                 # |lateral-offset slope|, MTW widths / s
    swerve_signed: float
    decel: float                  # relative drop of the depth-rate proxy, NaN if undefined
    clearance: float              # dx at the encounter (or window minimum if no crossing)
    clearance_from_crossing: bool
    encounter: bool               # True on the frame where dy crosses 0
    encounter_frame: int          # -1 if none yet
    direction: str                # "away" (dy: + -> -), "oncoming" (dy: - -> +), or "" before a crossing
    size_ratio: float
    size_class: str
    size_source: str              # "ratio" | "area_fraction"
    n_samples: int
    insufficient_history: bool
    standstill: bool
    ego_invalid: bool
    pothole_last_known: bool
    no_crossing: bool

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAssessment:
    frame: int
    t: float
    mtw_id: int
    pothole_id: int
    p_score: float
    t_score: float
    s_score: float
    d_score: float
    z_score: float
    total: float
    raw_label: str                # per-frame label before persistence / hysteresis
    gate_ok: bool
    reason: str
    features: Optional[InteractionFeatures] = field(default=None, repr=False)


@dataclass
class Event:
    event_id: str
    mtw_id: int
    pothole_id: int
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    n_frames: int
    peak_class: str
    peak_score: float
    peak_frame: int
    peak_t: float
    reason: str
    encountered: bool
    clearance: float
    clearance_from_crossing: bool
    direction: str
    size_class: str
    p_score: float
    t_score: float
    s_score: float
    d_score: float
    z_score: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)
