"""Re-run risk scoring + events from a saved interactions.csv (no video, no detectors).

Used by scripts/sensitivity_analysis.py and the tests; with the unchanged config it reproduces events.csv.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import pandas as pd

from ..config import RiskModeCfg
from ..schemas import Event, InteractionFeatures
from .events import EventAggregator
from .risk import RiskModel

_FIELDS = [f.name for f in dataclasses.fields(InteractionFeatures)]
_INT = {"frame", "mtw_id", "pothole_id", "encounter_frame", "n_samples"}
_BOOL = {"clearance_from_crossing", "encounter", "insufficient_history", "standstill", "ego_invalid",
         "pothole_last_known", "no_crossing"}
_STR = {"direction", "size_class", "size_source"}


def _val(name: str, v: Any) -> Any:
    if name in _STR:
        return "" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)
    if name in _INT:
        return int(v)
    if name in _BOOL:
        return str(v).strip().lower() in ("true", "1")
    return float(v) if v is not None and not (isinstance(v, str) and v == "") else math.nan


def features_from_frame(df: pd.DataFrame) -> list[InteractionFeatures]:
    return [InteractionFeatures(**{n: _val(n, r[n]) for n in _FIELDS}) for r in df.to_dict("records")]


def rescore(df: pd.DataFrame, block: RiskModeCfg, fps: float) -> list[Event]:
    """Events for the rows of ONE clip/run (rows must come from a single video)."""
    model = RiskModel(block)
    agg = EventAggregator(block, fps)
    feats = features_from_frame(df)
    by_frame: dict[int, list[InteractionFeatures]] = {}
    for f in feats:
        by_frame.setdefault(f.frame, []).append(f)
    if not by_frame:
        return []
    last = max(by_frame)
    for fr in range(min(by_frame), last + 1):
        agg.update(fr, [model.assess(f) for f in by_frame.get(fr, [])])
    return agg.finalize()
