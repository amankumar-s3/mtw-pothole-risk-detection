import math

from ptr.analysis.events import EventAggregator
from ptr.schemas import RiskAssessment
from test_risk import feat


def A(frame, label, mtw=1, poth=1, total=0.5, fps=10.0, **fk):
    f = feat(frame=frame, t=frame / fps, mtw_id=mtw, pothole_id=poth, **fk)
    return RiskAssessment(frame, frame / fps, mtw, poth, 0.5, 0.0, 0.5, 0.0, 0.6, total, label, label == "High",
                          f"{label} reason", f)


def agg(risk_cfg, fps=10.0):
    return EventAggregator(risk_cfg.for_mode("moving"), fps)      # accept 0.3 s = 3 frames, de-escalate 1.0 s = 10


def test_blip_is_not_accepted(risk_cfg):
    ag = agg(risk_cfg)
    labels = ["Low"] * 3 + ["High"] * 2 + ["Low"] * 3
    acc = [ag.update(i, [A(i, l)])[(1, 1)] for i, l in enumerate(labels)]
    assert set(acc) == {"Low"}


def test_escalation_after_persistence_and_slow_deescalation(risk_cfg):
    ag = agg(risk_cfg)
    labels = ["Low"] * 2 + ["High"] * 5 + ["Low"] * 12
    acc = [ag.update(i, [A(i, l)])[(1, 1)] for i, l in enumerate(labels)]
    assert acc[2:4] == ["Low", "Low"] and acc[4] == "High"        # accepted on the 3rd consecutive High
    assert acc[7 + 8] == "High"                                   # still High 9 frames after dropping
    assert acc[-1] == "Low"                                       # dropped after 10 low frames


def test_escalate_to_lowest_persisting_level(risk_cfg):
    ag = agg(risk_cfg)
    labels = ["Moderate", "High", "Moderate", "Moderate"]
    acc = [ag.update(i, [A(i, l)])[(1, 1)] for i, l in enumerate(labels)]
    assert acc[2] == "Moderate"                                   # 3 frames above Low, min level was Moderate


def test_event_peak_reason_and_grouping(risk_cfg):
    ag = agg(risk_cfg)
    for i in range(12):
        lab = "High" if 4 <= i <= 9 else "Low"
        ag.update(i, [A(i, lab, total=0.9 if lab == "High" else 0.1),
                      A(i, "Low", mtw=2, poth=1, total=0.05)])
    ev = ag.finalize()
    assert len(ev) == 2
    e1 = next(e for e in ev if e.mtw_id == 1)
    assert e1.peak_class == "High" and e1.peak_score == 0.9
    assert e1.reason.startswith("High:") and len(e1.reason) > 8
    e2 = next(e for e in ev if e.mtw_id == 2)
    assert e2.peak_class == "Low"
    s = EventAggregator.summarize(ev)
    assert s["events_per_class"] == {"Low": 1, "Moderate": 0, "High": 1} and s["n_events"] == 2


def test_short_encounters_are_discarded(risk_cfg):
    ag = agg(risk_cfg)
    ag.update(0, [A(0, "Low")])
    ag.update(1, [A(1, "Low")])
    assert ag.finalize() == [] and ag.n_discarded_short == 1


def test_event_closes_after_gap_and_reopens_as_new_event(risk_cfg):
    ag = agg(risk_cfg)
    for i in range(5):
        ag.update(i, [A(i, "Low")])
    for i in range(5, 30):                                        # no rows: > event_close_seconds
        ag.update(i, [])
    for i in range(30, 36):
        ag.update(i, [A(i, "Low")])
    ev = ag.finalize()
    assert len(ev) == 2 and ev[0].event_id != ev[1].event_id


def test_encounter_fields_in_event(risk_cfg):
    ag = agg(risk_cfg)
    for i in range(6):
        ag.update(i, [A(i, "Low", no_crossing=False, clearance=0.1, clearance_from_crossing=True,
                        direction="away", encounter_frame=2)])
    e = ag.finalize()[0]
    assert e.encountered and e.direction == "away" and math.isclose(e.clearance, 0.1)
