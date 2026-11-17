"""Video reading / writing. Timestamps are frame_index / fps by default (variable-fps safe alternative: msec)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..config import VideoCfg
from ..logging_setup import get_logger

log = get_logger("video")


@dataclass
class VideoInfo:
    path: str
    fps: float
    fps_from_file: bool
    width: int
    height: int
    n_frames: int


@dataclass
class FrameItem:
    frame: np.ndarray       # BGR, already resized if requested
    index: int              # processed-frame index (0, 1, 2, ...)
    video_index: int        # index in the original video
    t: float                # seconds
    scale: float            # resize factor applied to the original frame


def probe(path: str | Path, fallback_fps: float) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    from_file = math.isfinite(fps) and fps > 1e-3
    if not from_file:
        log.warning("video reports fps=%s; using fallback_fps=%s", fps, fallback_fps)
        fps = fallback_fps
    info = VideoInfo(str(path), fps, from_file, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    cap.release()
    return info


def output_size(info: VideoInfo, cfg: VideoCfg) -> tuple[int, int, float]:
    """(width, height, scale) after the optional resize."""
    if cfg.resize_width and cfg.resize_width != info.width:
        s = cfg.resize_width / info.width
        return cfg.resize_width, int(round(info.height * s)), s
    return info.width, info.height, 1.0


def iter_frames(path: str | Path, cfg: VideoCfg, info: VideoInfo | None = None) -> Iterator[FrameItem]:
    info = info or probe(path, cfg.fallback_fps)
    w, h, scale = output_size(info, cfg)
    cap = cv2.VideoCapture(str(path))
    vidx, k, last_t = -1, 0, -1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vidx += 1
            if vidx % cfg.frame_stride != 0:
                continue
            if cfg.max_frames is not None and k >= cfg.max_frames:
                break
            if cfg.timestamp_mode == "msec":
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if t <= last_t:                      # non-monotonic container timestamps: fall back
                    t = vidx / info.fps
            else:
                t = vidx / info.fps
            last_t = t
            if scale != 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            yield FrameItem(frame, k, vidx, t, scale)
            k += 1
    finally:
        cap.release()


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], fourcc: str) -> None:
        self.path = str(path)
        self.size = size
        self._w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if not self._w.isOpened():
            raise RuntimeError(f"cannot open video writer: {path} (fourcc={fourcc})")

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError("frame size changed")
        self._w.write(frame)

    def close(self) -> None:
        self._w.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *a) -> None:
        self.close()
