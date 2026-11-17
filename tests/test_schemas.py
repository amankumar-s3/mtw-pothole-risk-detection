import math

from ptr.schemas import (Detection, Event, InteractionFeatures, PotholeObs, RISK_LABELS, RiskAssessment,
                         Track, label_rank)


def test_label_rank_order():
    assert [label_rank(x) for x in RISK_LABELS] == [0, 1, 2]


def test_dataclasses_construct_and_serialise():
    d = Detection((0, 0, 10, 10), 0.9, 3, track_id=4)
    assert d.track_id == 4
    t = Track(1, 0, 0.0, (0, 0, 1, 1), (0, 0, 1, 1), 0.5, (0.5, 1.0))
    assert math.isnan(t.speed)
    p = PotholeObs(1, 0, 0.0, (0, 0, 1, 1), 0.5, True, False, 0, 3, "small")
    assert p.confirmed
    f = InteractionFeatures(0, 0.0, 1, 2, 0.1, 0.2, 0.22, 0.22, math.nan, 0.0, 0.0, math.nan, 0.0, False,
                            False, -1, "", 0.5, "medium", "ratio", 6, False, False, False, False, True)
    row = f.to_row()
    assert row["mtw_id"] == 1 and row["size_class"] == "medium"
    a = RiskAssessment(0, 0.0, 1, 2, 0, 0, 0, 0, 0, 0, "Low", False, "x")
    assert a.raw_label == "Low"
    e = Event("E1", 1, 2, 0, 5, 0.0, 0.2, 6, "Low", 0.1, 3, 0.1, "r", False, math.nan, False, "", "small",
              0, 0, 0, 0, 0)
    assert e.to_row()["event_id"] == "E1"
