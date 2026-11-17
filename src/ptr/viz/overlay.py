"""Annotated-frame drawing (OpenCV only). Never changes the frame size."""
from __future__ import annotations

import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..config import OutputCfg
from ..schemas import PotholeObs, Track, label_rank

COLORS = {            # BGR
    "Low": (70, 190, 70),
    "Moderate": (0, 165, 255),
    "High": (40, 40, 230),
    "none": (200, 160, 60),
}
POTHOLE_CONFIRMED = (255, 200, 0)
POTHOLE_TENTATIVE = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class ActivePair:
    mtw_id: int
    pothole_id: int
    label: str                       # accepted (post-hysteresis) label
    score: float
    reason: str
    mtw_contact: tuple[float, float]
    pothole_bbox: tuple[float, float, float, float]


@dataclass
class TrailStore:
    maxlen: int
    pts: dict[int, deque] = field(default_factory=dict)

    def add(self, tid: int, p: tuple[float, float]) -> None:
        self.pts.setdefault(tid, deque(maxlen=self.maxlen)).append((int(p[0]), int(p[1])))


def _text_bg(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _dashed_rect(img: np.ndarray, b: tuple[int, int, int, int], color, step: int = 6) -> None:
    x1, y1, x2, y2 = b
    for x in range(x1, x2, step * 2):
        cv2.line(img, (x, y1), (min(x + step, x2), y1), color, 1)
        cv2.line(img, (x, y2), (min(x + step, x2), y2), color, 1)
    for y in range(y1, y2, step * 2):
        cv2.line(img, (x1, y), (x1, min(y + step, y2)), color, 1)
        cv2.line(img, (x2, y), (x2, min(y + step, y2)), color, 1)


def render(frame: np.ndarray, tracks: list[Track], potholes: list[PotholeObs], pairs: list[ActivePair],
           trails: TrailStore, cfg: OutputCfg, frame_index: int, t: float,
           ego_valid: Optional[bool] = None, camera_mode: str = "moving") -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]
    fs = cfg.font_scale

    for p in potholes:
        b = tuple(int(round(v)) for v in p.bbox)
        if p.confirmed:
            cv2.rectangle(img, b[:2], b[2:], POTHOLE_CONFIRMED, 1 if p.last_known else 2)
            _text_bg(img, f"P{p.pothole_id} {p.size_class}", (b[0], max(12, b[1] - 3)), fs * 0.9)
        else:
            _dashed_rect(img, b, POTHOLE_TENTATIVE)

    worst: dict[int, str] = {}
    for pr in pairs:
        if pr.mtw_id not in worst or label_rank(pr.label) > label_rank(worst[pr.mtw_id]):
            worst[pr.mtw_id] = pr.label

    for tr in tracks:
        trails.add(tr.track_id, tr.contact)
        pts = list(trails.pts[tr.track_id])
        for i in range(1, len(pts)):
            a = i / len(pts)
            c = tuple(int(v * (0.3 + 0.7 * a)) for v in COLORS["none"])
            cv2.line(img, pts[i - 1], pts[i], c, 1)
        col = COLORS[worst.get(tr.track_id, "none")]
        b = tuple(int(round(v)) for v in tr.sbbox)
        cv2.rectangle(img, b[:2], b[2:], col, 2)
        tag = f"MTW {tr.track_id}" + ("" if tr.reliable else "?")
        _text_bg(img, tag, (b[0], max(12, b[1] - 3)), fs * 0.9, col)

    for pr in pairs:
        p0 = (int(pr.mtw_contact[0]), int(pr.mtw_contact[1]))
        pb = pr.pothole_bbox
        p1 = (int((pb[0] + pb[2]) / 2), int((pb[1] + pb[3]) / 2))
        cv2.line(img, p0, p1, COLORS[pr.label], 1, cv2.LINE_AA)

    # top-left panel: highest-risk active pair (semi-transparent, small)
    line_h = max(12, int(22 * fs))
    lines: list[str] = []
    top_label = None
    if pairs:
        top = max(pairs, key=lambda x: (label_rank(x.label), x.score))
        top_label = top.label
        lines = [f"MTW {top.mtw_id} / P{top.pothole_id}: {top.label.upper()} ({top.score:.2f})"]
        lines += textwrap.wrap(top.reason, width=max(30, int(W / (11 * fs))))[:3]
    panel_h = min(H, 10 + line_h * (1 + len(lines)))
    panel_w = min(W, int(W * 0.62))
    roi = img[0:panel_h, 0:panel_w]
    dark = np.zeros_like(roi)
    img[0:panel_h, 0:panel_w] = cv2.addWeighted(dark, 0.45, roi, 0.55, 0)
    cv2.putText(img, f"frame {frame_index}  t={t:.2f}s  mode={camera_mode}", (6, line_h - 2), FONT, fs * 0.9,
                (255, 255, 255), 1, cv2.LINE_AA)
    for i, ln in enumerate(lines):
        col = COLORS[top_label] if i == 0 and top_label else (255, 255, 255)
        cv2.putText(img, ln, (6, line_h * (i + 2) - 2), FONT, fs, col, 1, cv2.LINE_AA)

    # legend + ego-motion validity (small)
    lx, ly = 6, H - 8
    for name in ("Low", "Moderate", "High"):
        cv2.rectangle(img, (lx, ly - 9), (lx + 9, ly), COLORS[name], -1)
        cv2.putText(img, name, (lx + 12, ly), FONT, fs * 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        tw = cv2.getTextSize(name, FONT, fs * 0.8, 1)[0][0]
        lx += 12 + tw + 12
    cv2.putText(img, "potential-risk indicator, not accident prediction", (lx + 10, ly), FONT, fs * 0.7,
                (220, 220, 220), 1, cv2.LINE_AA)
    if cfg.draw_ego_indicator and ego_valid is not None:
        cv2.circle(img, (W - 12, 12), 5, (0, 200, 0) if ego_valid else (0, 0, 220), -1)
        cv2.putText(img, "ego", (W - 40, 16), FONT, fs * 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img
