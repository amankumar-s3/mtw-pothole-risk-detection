import dataclasses
import math

import pytest

from ptr.analysis.risk import RiskModel, linear_indicator
from ptr.config import Anchor
from ptr.schemas import InteractionFeatures


def feat(prox=5.0, ttr=math.nan, swerve=0.0, decel=math.nan, size="small", insufficient=False, **kw):
    base = dict(frame=0, t=0.0, mtw_id=1, pothole_id=1, dy=0.0, dx=0.0, dist=prox, proximity=prox, ttr=ttr,
                swerve=swerve, swerve_signed=swerve, decel=decel, clearance=0.0, clearance_from_crossing=False,
                encounter=False, encounter_frame=-1, direction="", size_ratio=math.nan, size_class=size,
                size_source="ratio", n_samples=10, insufficient_history=insufficient, standstill=False,
                ego_invalid=False, pothole_last_known=False, no_crossing=True)
    base.update(kw)
    return InteractionFeatures(**base)


def test_linear_indicator_both_directions_and_nan():
    a = Anchor(zero_at=4.0, one_at=1.0)
    assert linear_indicator(4.0, a) == 0.0 and linear_indicator(1.0, a) == 1.0
    assert linear_indicator(10.0, a) == 0.0 and linear_indicator(0.0, a) == 1.0
    assert math.isclose(linear_indicator(2.5, a), 0.5)
    assert linear_indicator(math.nan, a) == 0.0
    b = Anchor(zero_at=0.2, one_at=1.0)
    assert math.isclose(linear_indicator(0.6, b), 0.5)


def model(risk_cfg, mode="moving"):
    return RiskModel(risk_cfg.for_mode(mode))


def test_golden_low(risk_cfg):
    a = model(risk_cfg).assess(feat(prox=5.0, swerve=0.0, size="small"))
    w = risk_cfg.for_mode("moving").weights
    assert math.isclose(a.total, w.size * 0.3)
    assert a.raw_label == "Low" and a.reason


def test_golden_moderate(risk_cfg):
    a = model(risk_cfg).assess(feat(prox=2.0, ttr=2.0, swerve=0.1, size="medium"))
    w = risk_cfg.for_mode("moving").weights
    exp = w.proximity * (2 / 3) + w.ttr * 0.5 + w.size * 0.6
    assert math.isclose(a.total, exp)
    assert a.raw_label == "Moderate"


def test_golden_high(risk_cfg):
    a = model(risk_cfg).assess(feat(prox=0.8, ttr=1.0, swerve=1.2, size="large"))
    assert math.isclose(a.total, 1.0)
    assert a.raw_label == "High" and a.gate_ok
    assert "close pass" in a.reason and "large pothole" in a.reason


def test_high_gate_blocks_high_without_swerve(risk_cfg):
    a = model(risk_cfg).assess(feat(prox=0.8, ttr=1.0, swerve=0.0, size="large"))
    assert a.total >= risk_cfg.moving.classes.high_min
    assert a.raw_label == "Moderate" and not a.gate_ok
    assert "High needs" in a.reason and "swerve" in a.reason


def test_high_gate_blocks_small_pothole(risk_cfg):
    a = model(risk_cfg).assess(feat(prox=0.5, ttr=0.5, swerve=2.0, size="small"))
    assert a.raw_label != "High" and not a.gate_ok


def test_decel_only_counts_when_enabled(risk_cfg):
    f = feat(prox=0.8, ttr=1.0, swerve=0.0, decel=0.5, size="large")
    assert model(risk_cfg, "moving").assess(f).d_score == 1.0
    moving = model(risk_cfg, "moving").assess(f)
    static = model(risk_cfg, "static").assess(f)
    assert not moving.gate_ok                          # decel weight 0 => cannot open the gate
    assert static.gate_ok and static.raw_label == "High"


def test_yaml_change_changes_outcome_without_code(risk_cfg):
    f = feat(prox=2.0, ttr=2.0, swerve=0.1, size="medium")
    base = model(risk_cfg).assess(f)
    assert base.raw_label == "Moderate"
    mv = dataclasses.replace(risk_cfg.moving, classes=dataclasses.replace(risk_cfg.moving.classes, low_max=0.5, high_min=0.9))
    cfg2 = dataclasses.replace(risk_cfg, moving=mv)
    assert model(cfg2).assess(f).raw_label == "Low"
    w2 = dataclasses.replace(risk_cfg.moving.weights, proximity=3.0)
    cfg3 = dataclasses.replace(risk_cfg, moving=dataclasses.replace(risk_cfg.moving, weights=w2))
    assert model(cfg3).assess(f).total != base.total


def test_camera_mode_switch_selects_other_block(risk_cfg):
    f = feat(prox=1.0, ttr=1.5, swerve=0.9, decel=0.4, size="large")
    a_m = model(risk_cfg, "moving").assess(f)
    a_s = model(risk_cfg, "static").assess(f)
    assert risk_cfg.for_mode("moving") != risk_cfg.for_mode("static")
    assert a_m.total != a_s.total
    with pytest.raises(Exception):
        risk_cfg.for_mode("flying")


def test_insufficient_history_is_low_with_reason(risk_cfg):
    a = model(risk_cfg).assess(feat(prox=0.1, swerve=2.0, size="large", insufficient=True))
    assert a.raw_label == "Low" and "insufficient" in a.reason
