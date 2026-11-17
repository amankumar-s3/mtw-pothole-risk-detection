"""Explainable risk scoring: linear indicators, weighted sum, gated High class, readable reason.

Output is a POTENTIAL-RISK INDICATOR for an MTW-pothole encounter, never an accident prediction.
All numbers come from configs/risk.yaml ("initial hypotheses, to be calibrated").
"""
from __future__ import annotations

import math

from ..config import Anchor, RiskModeCfg
from ..schemas import InteractionFeatures, RiskAssessment


def linear_indicator(value: float, anchor: Anchor) -> float:
    """0 at ``zero_at``, 1 at ``one_at``, linear in between, clipped; NaN -> 0 (indicator unavailable)."""
    if value is None or not math.isfinite(value):
        return 0.0
    x = (value - anchor.zero_at) / (anchor.one_at - anchor.zero_at)
    return min(1.0, max(0.0, x))


class RiskModel:
    def __init__(self, block: RiskModeCfg) -> None:
        self.b = block

    @property
    def decel_enabled(self) -> bool:
        return self.b.weights.decel > 0.0

    def scores(self, f: InteractionFeatures) -> tuple[float, float, float, float, float]:
        a = self.b.anchors
        p = linear_indicator(f.proximity, a.proximity)
        t = linear_indicator(f.ttr, a.ttr)
        s = linear_indicator(f.swerve, a.swerve)
        d = linear_indicator(f.decel, a.decel)
        z = getattr(self.b.size_scores, f.size_class)
        return p, t, s, d, z

    def assess(self, f: InteractionFeatures) -> RiskAssessment:
        b = self.b
        p, t, s, d, z = self.scores(f)
        w = b.weights
        total = w.proximity * p + w.ttr * t + w.swerve * s + w.decel * d + w.size * z
        g = b.high_gate
        motion_ok = s >= g.s_min or (self.decel_enabled and d >= g.b_min)
        gate_ok = p >= g.p_min and motion_ok and z >= g.z_min
        if f.insufficient_history:
            label = "Low"
            reason = "insufficient track history, no motion features yet"
            return RiskAssessment(f.frame, f.t, f.mtw_id, f.pothole_id, p, t, s, d, z, total, label, False, reason, f)
        if total < b.classes.low_max:
            label = "Low"
        elif total < b.classes.high_min:
            label = "Moderate"
        else:
            label = "High" if gate_ok else "Moderate"
        reason = self._reason(f, label, total, p, s, d, z, gate_ok)
        return RiskAssessment(f.frame, f.t, f.mtw_id, f.pothole_id, p, t, s, d, z, total, label, gate_ok, reason, f)

    def _reason(self, f: InteractionFeatures, label: str, total: float, p: float, s: float, d: float,
                z: float, gate_ok: bool) -> str:
        g = self.b.high_gate
        parts = []
        if p >= g.p_min:
            parts.append(f"close pass (distance {f.proximity:.2f} MTW widths)")
        elif p > 0:
            parts.append(f"moderate distance ({f.proximity:.2f} MTW widths)")
        else:
            parts.append(f"far ({f.proximity:.2f} MTW widths)")
        if math.isfinite(f.swerve):
            if s >= g.s_min:
                parts.append(f"strong lateral shift {f.swerve:.2f} widths/s")
            elif s > 0:
                parts.append(f"mild lateral shift {f.swerve:.2f} widths/s")
            else:
                parts.append("no lateral shift")
        if self.decel_enabled and math.isfinite(f.decel) and d > 0:
            parts.append(f"speed drop {f.decel * 100:.0f}%")
        parts.append(f"{f.size_class} pothole")
        if math.isfinite(f.ttr) and self.b.weights.ttr > 0 and f.ttr < self.b.anchors.ttr.zero_at:
            parts.append(f"time-to-reach {f.ttr:.1f}s")
        text = ", ".join(parts) + f"; score {total:.2f}"
        if total >= self.b.classes.high_min and not gate_ok:
            missing = []
            if p < g.p_min:
                missing.append("proximity")
            if not (s >= g.s_min or (self.decel_enabled and d >= g.b_min)):
                missing.append("swerve" if not self.decel_enabled else "swerve/deceleration")
            if z < g.z_min:
                missing.append("pothole size")
            text += f" (High needs: {', '.join(missing)} above its gate)"
        return text
