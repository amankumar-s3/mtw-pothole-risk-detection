"""Tiny synthetic datasets for the data-script tests (no downloads)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_img(path: Path, seed: int, size=(96, 72), noise: float = 0.0) -> None:
    """Low-frequency random image: different seeds give very different perceptual hashes."""
    rng = np.random.default_rng(seed)
    low = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)
    img = cv2.resize(low, size, interpolation=cv2.INTER_CUBIC)
    if noise:
        img = np.clip(img.astype(float) + np.random.default_rng(seed + 999).normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def write_voc(path: Path, w: int, h: int, objs) -> None:
    body = "".join(f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
                   for n, a, b, c, d in objs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<annotation><size><width>{w}</width><height>{h}</height></size>{body}</annotation>")


def write_yolo(path: Path, rows, crlf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nl = "\r\n" if crlf else "\n"
    path.write_text("".join(f"{c} {x} {y} {w} {h}{nl}" for c, x, y, w, h in rows))


def make_voc_source(root: Path, prefix: str, n: int, seed0: int, cls: str = "D40") -> None:
    for i in range(n):
        make_img(root / "train" / "images" / f"{prefix}_{i:03d}.jpg", seed0 + i)
        write_voc(root / "train" / "annotations" / "xmls" / f"{prefix}_{i:03d}.xml", 96, 72,
                  [(cls, 10, 10, 50, 40)])


def make_yolo_source(root: Path, prefix: str, n: int, seed0: int) -> None:
    for i in range(n):
        make_img(root / "images" / f"{prefix}_{i:03d}.jpg", seed0 + i)
        write_yolo(root / "images" / f"{prefix}_{i:03d}.txt", [(0, 0.5, 0.5, 0.4, 0.4)])
