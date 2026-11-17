"""System-vs-human agreement on Low / Moderate / High, plus inter-rater agreement. Always reports the sample size.

This measures agreement with human judgement on the video, NOT the correctness of any accident prediction.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ..schemas import RISK_LABELS

IDX = {l: i for i, l in enumerate(RISK_LABELS)}


def _to_idx(labels: Sequence[str]) -> list[int]:
    out = []
    for l in labels:
        k = str(l).strip().capitalize()
        if k not in IDX:
            raise ValueError(f"unknown risk label {l!r}; expected one of {list(RISK_LABELS)}")
        out.append(IDX[k])
    return out


def confusion(reference: Sequence[str], system: Sequence[str]) -> np.ndarray:
    """3x3 counts; rows = reference (human), columns = system."""
    m = np.zeros((3, 3), dtype=int)
    for r, s in zip(_to_idx(reference), _to_idx(system)):
        m[r, s] += 1
    return m


def weighted_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Linearly weighted Cohen's kappa over the ordered labels; NaN if undefined (e.g. a single constant label)."""
    ia, ib = _to_idx(a), _to_idx(b)
    if len(ia) == 0:
        return math.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # sklearn warns when kappa is undefined (constant labels)
        k = cohen_kappa_score(ia, ib, labels=[0, 1, 2], weights="linear")
    return float(k) if math.isfinite(k) else math.nan


def agreement_report(reference: Sequence[str], system: Sequence[str]) -> dict:
    n = len(reference)
    cm = confusion(reference, system)
    per_class = {}
    for i, lab in enumerate(RISK_LABELS):
        tp = int(cm[i, i])
        sys_n, ref_n = int(cm[:, i].sum()), int(cm[i, :].sum())
        per_class[lab] = {"precision": tp / sys_n if sys_n else math.nan, "recall": tp / ref_n if ref_n else math.nan,
                          "n_reference": ref_n, "n_system": sys_n}
    return {"n": n, "accuracy": float(np.trace(cm) / n) if n else math.nan, "weighted_kappa": weighted_kappa(reference, system),
            "confusion_reference_rows_system_cols": cm.tolist(), "per_class": per_class}


def inter_rater(r1: Sequence[str], r2: Sequence[str]) -> dict:
    n = len(r1)
    ia, ib = _to_idx(r1), _to_idx(r2)
    return {"n": n, "weighted_kappa": weighted_kappa(r1, r2),
            "exact_agreement": float(np.mean([a == b for a, b in zip(ia, ib)])) if n else math.nan}


def load_labelled_events(events_csv: Path, labels_csv: Path, rater_cols: Sequence[str] = ("rater1", "rater2"),
                         consensus_col: Optional[str] = "consensus") -> pd.DataFrame:
    """Join events.csv with the human labels (key: event_id [+ clip_id]); drops events without complete labels."""
    ev = pd.read_csv(events_csv)
    lab = pd.read_csv(labels_csv)
    keys = ["event_id"] + (["clip_id"] if "clip_id" in lab.columns and "clip_id" in ev.columns else [])
    df = ev.merge(lab[keys + [c for c in list(rater_cols) + ([consensus_col] if consensus_col else []) if c in lab.columns]],
                  on=keys, how="inner")
    need = [c for c in rater_cols if c in df.columns]
    df = df.dropna(subset=need)
    df = df[(df[need].astype(str).apply(lambda col: col.str.strip() != "")).all(axis=1)]
    return df.reset_index(drop=True)


def split_by_clip(clip_ids: Sequence[str], calibration_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Split CLIPS (never events) into calibration and held-out sets; deterministic for a seed."""
    import random

    clips = sorted(set(clip_ids))
    rng = random.Random(seed)
    rng.shuffle(clips)
    k = int(round(calibration_fraction * len(clips)))
    if len(clips) >= 2:
        k = min(max(k, 1), len(clips) - 1)            # both sets non-empty when possible
    return sorted(clips[:k]), sorted(clips[k:])
