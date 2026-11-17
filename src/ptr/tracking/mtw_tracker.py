"""Model B: COCO-pretrained YOLOv8s filtered to the motorcycle class + Ultralytics' built-in ByteTrack."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import MtwDetCfg, resolve_path
from ..detection.detector import _need_ultralytics, filter_min_size
from ..logging_setup import get_logger
from ..schemas import Detection

log = get_logger("mtw_tracker")


class MTWTracker:
    def __init__(self, cfg: MtwDetCfg) -> None:
        self.cfg = cfg
        YOLO = _need_ultralytics()
        self.model = YOLO(resolve_path(cfg.weights))          # bare names are auto-downloaded by Ultralytics
        self.tracker_cfg = resolve_path(cfg.tracker_config)
        if not Path(self.tracker_cfg).is_file():
            raise FileNotFoundError(f"tracker config not found: {self.tracker_cfg}")
        log.info("MTW detector loaded: %s, tracker: %s", cfg.weights, self.tracker_cfg)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """One frame. ``persist=True`` keeps ByteTrack state between calls (frame-by-frame use)."""
        c = self.cfg
        res = self.model.track(frame, persist=True, tracker=self.tracker_cfg, classes=[c.coco_class_id],
                               conf=c.conf, iou=c.iou, imgsz=c.imgsz, device=c.device, verbose=False)[0]
        out: list[Detection] = []
        b = res.boxes
        if b is not None and len(b) and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            for box, s, i in zip(xyxy, conf, ids):
                out.append(Detection((float(box[0]), float(box[1]), float(box[2]), float(box[3])), float(s),
                                     c.coco_class_id, int(i)))
        return filter_min_size(out, c.min_box_px)
