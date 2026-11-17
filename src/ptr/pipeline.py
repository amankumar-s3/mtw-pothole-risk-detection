"""Pipeline: orchestrates detection, tracking, ego-motion, features, risk, events, overlay and outputs.

Writes outputs/<run_id>/{annotated.mp4, tracks.csv, potholes.csv, interactions.csv, events.csv, summary.json,
egomotion.csv, config_used.yaml, run.log}. The output is a potential-risk indicator, not accident prediction.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import yaml

from .analysis.egomotion import EgoHistory, EgoMotionEstimator, identity_result
from .analysis.events import EventAggregator
from .analysis.features import FeatureExtractor
from .analysis.pothole_registry import PotholeRegistry
from .analysis.risk import RiskModel
from .config import PipelineConfig, RiskConfig, to_dict
from .detection.detector import DetectionSource
from .io.video import VideoWriter, iter_frames, output_size, probe
from .logging_setup import get_logger, make_run_id, setup_logging
from .schemas import InteractionFeatures, RISK_LABELS
from .utils import ensure_dir, write_csv
from .viz.overlay import ActivePair, TrailStore, render

log = get_logger("pipeline")

STATUS = "potential-risk indicator for MTW-pothole encounters; NOT accident prediction"

TRACK_COLS = ["clip_id", "frame", "t", "track_id", "x1", "y1", "x2", "y2", "sx1", "sy1", "sx2", "sy2", "conf",
              "contact_x", "contact_y", "vx", "vy", "speed", "heading_deg", "age_frames", "reliable"]
POTHOLE_COLS = ["clip_id", "frame", "t", "pothole_id", "x1", "y1", "x2", "y2", "conf", "confirmed", "last_known",
                "last_seen_frame", "hits", "size_class"]
EGO_COLS = ["frame", "t", "valid", "reason", "n_points", "inlier_ratio", "tx", "ty", "rot_deg", "scale"]
RISK_COLS = ["p_score", "t_score", "s_score", "d_score", "z_score", "total", "raw_label", "gate_ok",
             "accepted_label", "reason"]
EVENT_COLS = ["event_id", "clip_id", "mtw_id", "pothole_id", "start_frame", "end_frame", "start_t", "end_t",
              "n_frames", "peak_class", "peak_score", "peak_frame", "peak_t", "reason", "encountered", "clearance",
              "clearance_from_crossing", "direction", "size_class", "p_score", "t_score", "s_score", "d_score",
              "z_score"]


def interaction_cols() -> list[str]:
    return ["clip_id"] + list(InteractionFeatures.__dataclass_fields__.keys()) + RISK_COLS


class Pipeline:
    def __init__(self, pipe_cfg: PipelineConfig, risk_cfg: RiskConfig, source: DetectionSource,
                 out_root: str | Path = "outputs", run_id: Optional[str] = None,
                 meta: Optional[dict[str, Any]] = None) -> None:
        self.cfg = pipe_cfg
        self.risk_cfg = risk_cfg
        self.source = source
        self.run_id = run_id or make_run_id()
        self.run_dir = ensure_dir(Path(out_root) / self.run_id)
        self.meta = meta or {}
        setup_logging(self.run_dir, self.run_id)

    # ------------------------------------------------------------------ run
    def run(self, video_path: str | Path) -> dict[str, Any]:
        cfg = self.cfg
        t_wall = time.perf_counter()
        clip_id = Path(video_path).stem
        info = probe(video_path, cfg.video.fallback_fps)
        W, H, scale = output_size(info, cfg.video)
        fps_eff = info.fps / cfg.video.frame_stride
        log.info("run %s | video=%s | %dx%d @ %.3f fps (stride %d) | camera_mode=%s", self.run_id, video_path,
                 W, H, info.fps, cfg.video.frame_stride, cfg.camera_mode)
        self._dump_config(video_path, info.fps, fps_eff, (W, H))

        block = self.risk_cfg.for_mode(cfg.camera_mode)
        risk = RiskModel(block)
        events = EventAggregator(block, fps_eff)
        from .tracking.trajectory import TrajectoryStore

        traj = TrajectoryStore(cfg.trajectory)
        reg = PotholeRegistry(cfg.registry, cfg.features.registry_size, cfg.camera_mode, fps_eff, (W, H))
        ego_hist = EgoHistory(math.ceil(cfg.ego_motion.max_span_seconds * fps_eff))
        ego_est = EgoMotionEstimator(cfg.ego_motion) if cfg.ego_motion.enabled else None
        feats = FeatureExtractor(cfg.features, fps_eff, (W, H), cfg.trajectory.min_history_frames,
                                 ego_hist if cfg.ego_motion.enabled else None)
        trails = TrailStore(cfg.output.trail_length)
        writer = VideoWriter(self.run_dir / "annotated.mp4", fps_eff, (W, H), cfg.video.writer_fourcc) \
            if cfg.output.write_video else None

        track_rows: list[dict] = []
        poth_rows: list[dict] = []
        ego_rows: list[dict] = []
        inter_rows: list[dict] = []
        prev_gray = None
        prev_boxes: list = []
        n = 0
        ego_valid_n = 0
        last_t = 0.0
        try:
            for fi in iter_frames(video_path, cfg.video, info):
                k, t, frame = fi.index, fi.t, fi.frame
                last_t = t
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pot_dets, mtw_dets = self.source.detect(frame, fi.video_index, fi.scale)
                tracks = traj.update(k, t, mtw_dets)

                if prev_gray is None:
                    ego = identity_result(k, "first_frame", valid=True)
                elif ego_est is not None:
                    ego = ego_est.estimate(k, prev_gray, gray, prev_boxes)
                else:
                    ego = identity_result(k, "disabled", valid=True)
                ego_hist.push(k, ego.M, ego.valid)
                ego_valid_n += int(ego.valid)
                ego_rows.append(ego.row(t))
                prev_gray = gray
                prev_boxes = [d.bbox for d in pot_dets] + [d.bbox for d in mtw_dets]

                potholes = reg.update(k, t, pot_dets, ego_hist if cfg.ego_motion.enabled else None)

                rows = feats.update(k, t, tracks, potholes)
                assess = [risk.assess(r) for r in rows]
                accepted = events.update(k, assess)

                for a in assess:
                    d = a.features.to_row()
                    d.update({"clip_id": clip_id, "p_score": a.p_score, "t_score": a.t_score,
                              "s_score": a.s_score, "d_score": a.d_score, "z_score": a.z_score, "total": a.total,
                              "raw_label": a.raw_label, "gate_ok": a.gate_ok,
                              "accepted_label": accepted[(a.mtw_id, a.pothole_id)], "reason": a.reason})
                    inter_rows.append(d)
                for tr in tracks:
                    track_rows.append({"clip_id": clip_id, "frame": k, "t": t, "track_id": tr.track_id,
                                       "x1": tr.bbox[0], "y1": tr.bbox[1], "x2": tr.bbox[2], "y2": tr.bbox[3],
                                       "sx1": tr.sbbox[0], "sy1": tr.sbbox[1], "sx2": tr.sbbox[2], "sy2": tr.sbbox[3],
                                       "conf": tr.conf, "contact_x": tr.contact[0], "contact_y": tr.contact[1],
                                       "vx": tr.vx, "vy": tr.vy, "speed": tr.speed, "heading_deg": tr.heading_deg,
                                       "age_frames": tr.age_frames, "reliable": int(tr.reliable)})
                for p in potholes:
                    poth_rows.append({"clip_id": clip_id, "frame": k, "t": t, "pothole_id": p.pothole_id,
                                      "x1": p.bbox[0], "y1": p.bbox[1], "x2": p.bbox[2], "y2": p.bbox[3],
                                      "conf": p.conf, "confirmed": int(p.confirmed), "last_known": int(p.last_known),
                                      "last_seen_frame": p.last_seen_frame, "hits": p.hits, "size_class": p.size_class})
                if writer is not None:
                    pot_by_id = {p.pothole_id: p for p in potholes}
                    tr_by_id = {tr.track_id: tr for tr in tracks}
                    active = []
                    for a in assess:
                        tr, p = tr_by_id.get(a.mtw_id), pot_by_id.get(a.pothole_id)
                        if tr is None:
                            continue
                        pb = p.bbox if p is not None else (0, 0, 0, 0)
                        if p is None and a.features is not None:
                            continue
                        active.append(ActivePair(a.mtw_id, a.pothole_id, accepted[(a.mtw_id, a.pothole_id)],
                                                 a.total, a.reason, tr.contact, pb))
                    writer.write(render(frame, tracks, potholes, active, trails, cfg.output, k, t,
                                        ego_valid=ego.valid if cfg.ego_motion.enabled else None,
                                        camera_mode=cfg.camera_mode))
                n += 1
                if n % 100 == 0:
                    log.info("processed %d frames (t=%.1fs)", n, t)
        finally:
            if writer is not None:
                writer.close()

        ev = events.finalize()
        lifetimes = traj.finalize()
        unreliable = sorted(i for i, life in lifetimes.items() if life < cfg.trajectory.min_reliable_seconds)
        ev_rows = []
        for e in ev:
            d = e.to_row()
            d["clip_id"] = clip_id
            ev_rows.append(d)
        write_csv(self.run_dir / "tracks.csv", track_rows, TRACK_COLS)
        write_csv(self.run_dir / "potholes.csv", poth_rows, POTHOLE_COLS)
        write_csv(self.run_dir / "egomotion.csv", ego_rows, EGO_COLS)
        write_csv(self.run_dir / "interactions.csv", inter_rows, interaction_cols())
        write_csv(self.run_dir / "events.csv", ev_rows, EVENT_COLS)

        wall = time.perf_counter() - t_wall
        summary = {
            "run_id": self.run_id, "clip_id": clip_id, "video": str(video_path), "status": STATUS,
            "camera_mode": cfg.camera_mode, "risk_status": self.risk_cfg.status,
            "mock_detections": bool(self.meta.get("mock_detections")),
            "video_fps": info.fps, "effective_fps": fps_eff, "frame_size": [W, H], "frames_processed": n,
            "duration_s": last_t,
            "n_mtw_tracks": len(lifetimes), "unreliable_mtw_track_ids": unreliable,
            "n_pothole_ids_seen": len({r["pothole_id"] for r in poth_rows}),
            "n_pothole_ids_confirmed": len({r["pothole_id"] for r in poth_rows if r["confirmed"]}),
            "registry_dropped": reg.n_dropped, "registry_merged": reg.n_merged,
            "ego_motion_enabled": cfg.ego_motion.enabled,
            "ego_motion_valid_fraction": (ego_valid_n / n) if n else None,
            "events_discarded_too_short": events.n_discarded_short,
            "wall_time_s": wall, "processing_fps_wall_clock": (n / wall) if wall > 0 else None,
            **EventAggregator.summarize(ev),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        log.info("done: %d frames, %d events %s, wall %.1fs", n, len(ev), summary["events_per_class"], wall)
        return summary

    def _dump_config(self, video: str | Path, fps: float, fps_eff: float, size: tuple[int, int]) -> None:
        doc = {"run": {"run_id": self.run_id, "video": str(video), "fps": fps, "effective_fps": fps_eff,
                       "frame_size": list(size), "camera_mode": self.cfg.camera_mode, **self.meta},
               "pipeline": to_dict(self.cfg),
               "risk": to_dict(self.risk_cfg),
               "risk_block_used": to_dict(self.risk_cfg.for_mode(self.cfg.camera_mode))}
        (self.run_dir / "config_used.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
