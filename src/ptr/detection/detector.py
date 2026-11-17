"""Detectors: single-class pothole YOLOv8 (model A) and a mock source for testing without weights.

The MTW detector+tracker (model B) is in tracking/mtw_tracker.py. The two models are never merged.
Ultralytics is imported lazily so that everything else (tests, mock mode) works without it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import PipelineConfig, PotholeDetCfg, resolve_path
from ..logging_setup import get_logger
from ..schemas import Detection

log = get_logger("detector")


def _need_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError("ultralytics is required for real detection (pip install ultralytics). "
                          "Use --mock-detections to run without it.") from exc
    return YOLO


def filter_min_size(dets: list[Detection], min_px: float) -> list[Detection]:
    return [d for d in dets if (d.bbox[2] - d.bbox[0]) >= min_px and (d.bbox[3] - d.bbox[1]) >= min_px]


class DetectionSource(Protocol):
    def detect(self, frame: np.ndarray, video_index: int, scale: float) -> tuple[list[Detection], list[Detection]]:
        """Return (pothole detections, MTW detections with track ids)."""
        ...


class PotholeDetector:
    """Model A: YOLOv8s fine-tuned on one class (pothole)."""

    def __init__(self, cfg: PotholeDetCfg) -> None:
        self.cfg = cfg
        YOLO = _need_ultralytics()
        w = resolve_path(cfg.weights)
        if not Path(w).is_file():
            raise FileNotFoundError(f"pothole weights not found: {w} (train with scripts/train_pothole.py "
                                    f"and copy best.pt there, or use --mock-detections)")
        self.model = YOLO(w)
        log.info("pothole detector loaded: %s", w)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        c = self.cfg
        res = self.model.predict(frame, conf=c.conf, iou=c.iou, imgsz=c.imgsz, max_det=c.max_det,
                                 device=c.device, verbose=False)[0]
        out: list[Detection] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            for b, s in zip(xyxy, conf):
                out.append(Detection((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(s), 0))
        return filter_min_size(out, c.min_box_px)


class RealSource:
    """Both real detectors behind the DetectionSource interface."""

    def __init__(self, cfg: PipelineConfig) -> None:
        from ..tracking.mtw_tracker import MTWTracker

        self.pothole = PotholeDetector(cfg.detectors.pothole)
        self.mtw = MTWTracker(cfg.detectors.mtw)

    def detect(self, frame: np.ndarray, video_index: int, scale: float):
        return self.pothole.detect(frame), self.mtw.track(frame)


class MockDetections:
    """Mock detections from JSON.

    {"fps": 30, "frames": [{"frame": 0,
        "potholes": [{"bbox_xyxy": [x1,y1,x2,y2], "conf": 0.9}],
        "mtws":     [{"track_id": 1, "bbox_xyxy": [x1,y1,x2,y2], "conf": 0.9}]}, ...]}

    ``frame`` is the index in the ORIGINAL video. Boxes are in original pixel coordinates (scaled with the
    pipeline's resize factor). MTWs carry track ids, so ByteTrack is bypassed in mock mode.
    """

    def __init__(self, doc: dict, min_pothole_px: float = 0.0, min_mtw_px: float = 0.0) -> None:
        if "frames" not in doc or not isinstance(doc["frames"], list):
            raise ValueError("mock detections JSON needs a 'frames' list")
        self.fps = doc.get("fps")
        self._by_frame: dict[int, dict] = {}
        for fr in doc["frames"]:
            self._by_frame[int(fr["frame"])] = fr
            for m in fr.get("mtws", []):
                if "track_id" not in m:
                    raise ValueError("every mock MTW needs a track_id")
        self.min_pothole_px = min_pothole_px
        self.min_mtw_px = min_mtw_px

    @classmethod
    def from_file(cls, path: str | Path, min_pothole_px: float = 0.0, min_mtw_px: float = 0.0) -> "MockDetections":
        return cls(json.loads(Path(path).read_text()), min_pothole_px, min_mtw_px)

    def detect(self, frame: np.ndarray, video_index: int, scale: float):
        fr = self._by_frame.get(video_index, {})
        pot = [Detection(tuple(c * scale for c in p["bbox_xyxy"]), float(p.get("conf", 1.0)), 0)  # type: ignore[arg-type]
               for p in fr.get("potholes", [])]
        mtw = [Detection(tuple(c * scale for c in m["bbox_xyxy"]), float(m.get("conf", 1.0)), 3,  # type: ignore[arg-type]
                         int(m["track_id"])) for m in fr.get("mtws", [])]
        return filter_min_size(pot, self.min_pothole_px), filter_min_size(mtw, self.min_mtw_px)
