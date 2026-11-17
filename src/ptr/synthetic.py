"""Synthetic pinhole ground-plane scenes (used by tests and scripts/make_synthetic_video.py).

World: X to the right, Z forward, camera at height ``cam_height`` looking straight ahead, moving forward at
``cam_speed``. A ground point (X, Z) at camera depth Zc = Z - cam_z projects to
    u = cx + f X / Zc,   v = horizon_row + f cam_height / Zc
so the image row grows as an object gets closer (the convention used in analysis/geometry.py).
All numbers here are made up for testing; they are NOT measurements of anything real.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .schemas import BBox

MTW_W = 0.7   # metres (test scene only)
MTW_H = 1.4
POTHOLE_MIN_ASPECT = 0.35  # box height / width of simulated pothole boxes (test scene only)


@dataclass
class CameraModel:
    width: int = 640
    height: int = 360
    f: float = 500.0
    horizon_row: float = 150.0
    cam_height: float = 1.2
    min_depth: float = 4.0

    @property
    def cx(self) -> float:
        return self.width / 2.0

    def project(self, X: float, Zc: float) -> tuple[float, float]:
        return self.cx + self.f * X / Zc, self.horizon_row + self.f * self.cam_height / Zc


@dataclass
class PotholeSpec:
    X: float
    Z: float
    rx: float = 0.45
    rz: float = 0.7


@dataclass
class MtwSpec:
    mtw_id: int
    Z0: float
    speed: float                              # world m/s along +Z (negative = oncoming)
    X_fn: Callable[[float], float]
    scenario: str = ""
    Z_fn: Optional[Callable[[float], float]] = None   # overrides Z0 + speed * (t - t_start) (e.g. braking)
    t_start: float = 0.0                              # the MTW appears at Z0 at this time


@dataclass
class Scene:
    cam: CameraModel
    fps: float
    duration: float
    cam_speed: float
    potholes: list[PotholeSpec]
    mtws: list[MtwSpec]
    dropout: float = 0.0                       # probability that the pothole detection is missing
    box_noise_px: float = 0.0
    seed: int = 0

    @property
    def n_frames(self) -> int:
        return int(round(self.fps * self.duration))


@dataclass
class FrameTruth:
    frame: int
    t: float
    mtws: dict[int, BBox] = field(default_factory=dict)
    potholes: dict[int, BBox] = field(default_factory=dict)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def mtw_box(cam: CameraModel, X: float, Zc: float) -> BBox:
    u, v = cam.project(X, Zc)
    w, h = cam.f * MTW_W / Zc, cam.f * MTW_H / Zc
    return (u - w / 2, v - h, u + w / 2, v)


def pothole_box(cam: CameraModel, p: PotholeSpec, cam_z: float) -> Optional[BBox]:
    if p.Z - p.rz - cam_z < cam.min_depth:
        return None
    th = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    us, vs = [], []
    for a in th:
        u, v = cam.project(p.X + p.rx * math.cos(a), p.Z + p.rz * math.sin(a) - cam_z)
        us.append(u)
        vs.append(v)
    x1, x2, y1, y2 = min(us), max(us), min(vs), max(vs)
    # A real detector box also covers rim and shadow: keep the bottom row, enforce a minimum aspect ratio.
    y1 = min(y1, y2 - POTHOLE_MIN_ASPECT * (x2 - x1))
    return (x1, y1, x2, y2)


def shake_sequence(n: int, amp_px: float, seed: int = 1) -> np.ndarray:
    """Absolute image shift (n, 2) of a bounded random walk; zero if amp_px == 0."""
    if amp_px <= 0:
        return np.zeros((n, 2))
    rng = np.random.default_rng(seed)
    s = np.zeros((n, 2))
    for i in range(1, n):
        s[i] = np.clip(0.8 * s[i - 1] + rng.normal(0, amp_px * 0.5, 2), -amp_px, amp_px)
    return s


def shake_to_ego(shake: np.ndarray) -> list[np.ndarray]:
    """Frame-to-frame 2x3 transforms (previous -> current) of an absolute shift sequence."""
    out = [np.array([[1.0, 0, 0], [0, 1.0, 0]])]
    for i in range(1, len(shake)):
        d = shake[i] - shake[i - 1]
        out.append(np.array([[1.0, 0, d[0]], [0, 1.0, d[1]]]))
    return out


def simulate(scene: Scene, shake: Optional[np.ndarray] = None, noisy: bool = True) -> list[FrameTruth]:
    """Per-frame boxes. ``shake`` (n, 2) is added to every box (shared camera shake)."""
    rng = np.random.default_rng(scene.seed)
    n = scene.n_frames
    shake = np.zeros((n, 2)) if shake is None else shake
    out: list[FrameTruth] = []
    for k in range(n):
        t = k / scene.fps
        cam_z = scene.cam_speed * t
        ft = FrameTruth(k, t)
        sx, sy = shake[k]
        for m in scene.mtws:
            if t < m.t_start:
                continue
            Zm = m.Z_fn(t) if m.Z_fn is not None else m.Z0 + m.speed * (t - m.t_start)
            Zc = Zm - cam_z
            if Zc < scene.cam.min_depth:
                continue
            b = mtw_box(scene.cam, m.X_fn(t), Zc)
            if noisy and scene.box_noise_px > 0:
                b = tuple(c + rng.normal(0, scene.box_noise_px) for c in b)  # type: ignore[assignment]
            ft.mtws[m.mtw_id] = (b[0] + sx, b[1] + sy, b[2] + sx, b[3] + sy)
        for i, p in enumerate(scene.potholes, start=1):
            b = pothole_box(scene.cam, p, cam_z)
            if b is None:
                continue
            if noisy and scene.dropout > 0 and rng.random() < scene.dropout:
                continue
            if noisy and scene.box_noise_px > 0:
                b = tuple(c + rng.normal(0, scene.box_noise_px) for c in b)  # type: ignore[assignment]
            ft.potholes[i] = (b[0] + sx, b[1] + sy, b[2] + sx, b[3] + sy)
        out.append(ft)
    return out


# --------------------------------------------------------------------------- named scenarios
def scenario(name: str, direction: str = "away", fps: float = 30.0, duration: float = 3.6,
             cam_speed: float = 6.0, pothole_X: float = 0.5, cam: Optional[CameraModel] = None,
             noise_px: float = 0.0, dropout: float = 0.0, seed: int = 0) -> Scene:
    """One MTW and one pothole. ``name``: far_pass | near_pass | through | late_swerve | early_swerve.

    The MTW reaches the pothole's depth at t = 2.2 s (world positions are chosen so that the crossing of
    dy happens there for both directions of travel)."""
    cam = cam or CameraModel()
    pz = 40.0
    speed = 9.0 if direction == "away" else -9.0
    z0 = pz - 2.2 * speed
    t_cross = 2.2
    px = pothole_X
    if name == "far_pass":
        xf = lambda t: px + 3.5                                  # noqa: E731
    elif name == "near_pass":
        xf = lambda t: px + 1.0                                  # noqa: E731
    elif name == "through":
        xf = lambda t: px                                        # noqa: E731
    elif name == "late_swerve":                                  # shifts 1.0 m sideways just before the pothole
        xf = lambda t: px + 1.0 * smoothstep((t - (t_cross - 1.2)) / 0.8)  # noqa: E731
    elif name == "early_swerve":                                 # same shift, long before the window of interest
        xf = lambda t: px + 1.0 * smoothstep((t - 0.2) / 0.5)    # noqa: E731
    else:
        raise ValueError(f"unknown scenario {name!r}")
    return Scene(cam=cam, fps=fps, duration=duration, cam_speed=cam_speed,
                 potholes=[PotholeSpec(px, pz)], mtws=[MtwSpec(1, z0, speed, xf, name)],
                 dropout=dropout, box_noise_px=noise_px, seed=seed)


def demo_scene(fps: float = 30.0, duration: float = 8.0, seed: int = 0, noise_px: float = 1.0,
               dropout: float = 0.08, cam: Optional[CameraModel] = None) -> Scene:
    """Three potholes and three MTWs (far pass, through, late swerve) for scripts/make_synthetic_video.py.

    Each MTW appears 8 m ahead of the camera, moves away at 7 m/s (camera 4 m/s) and reaches its pothole's
    depth 2.2 s later, at about 15 m camera depth. The three encounters happen at t = 2.2, 4.4 and 6.6 s."""
    cam = cam or CameraModel(width=960, height=540, f=750.0, horizon_row=225.0)
    cs, sp, tau = 4.0, 7.0, 2.2
    starts = [0.0, 2.2, 4.4]
    px = [-1.2, 0.8, -0.5]
    potholes, mtws = [], []
    for i, (ts, x) in enumerate(zip(starts, px), start=1):
        z0 = cs * ts + 8.0
        potholes.append(PotholeSpec(x, z0 + sp * tau))
        tc = ts + tau
        if i == 1:
            xf = (lambda x: (lambda t: x + 3.5))(x)
            name = "far_pass"
        elif i == 2:
            xf = (lambda x: (lambda t: x))(x)
            name = "through"
        else:
            xf = (lambda x, tc: (lambda t: x + 1.0 * smoothstep((t - (tc - 1.2)) / 0.8)))(x, tc)
            name = "late_swerve"
        mtws.append(MtwSpec(i, z0, sp, xf, name, t_start=ts))
    return Scene(cam=cam, fps=fps, duration=duration, cam_speed=cs, potholes=potholes, mtws=mtws,
                 dropout=dropout, box_noise_px=noise_px, seed=seed)


# --------------------------------------------------------------------------- rendering + mock export
def render_video(scene: Scene, out_path: Path, shake: np.ndarray, truth: list[FrameTruth],
                 fourcc: str = "mp4v") -> None:
    """Render a simple textured road video that matches ``truth`` (boxes already include the shake)."""
    import cv2

    cam = scene.cam
    W, H = cam.width, cam.height
    rng = np.random.default_rng(scene.seed + 7)
    n_dots = 2500
    dots = np.stack([rng.uniform(-9, 9, n_dots), rng.uniform(2, 140, n_dots), rng.integers(30, 230, n_dots)], 1)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), scene.fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {out_path}")
    for k, ft in enumerate(truth):
        cam_z = scene.cam_speed * ft.t
        sx, sy = shake[k]
        img = np.zeros((H, W, 3), np.uint8)
        hr = int(round(cam.horizon_row + sy))
        img[:max(hr, 0)] = (200, 170, 120)
        for y in range(max(hr, 0), H):
            g = 90 + int(60 * (y - hr) / max(1, H - hr))
            img[y] = (g, g, g)
        # road edges
        for X in (-3.6, 3.6):
            pts = []
            for Zc in (4.0, 60.0, 300.0):
                u, v = cam.project(X, Zc)
                pts.append((int(u + sx), int(v + sy)))
            cv2.line(img, pts[0], pts[2], (230, 230, 230), 2)
        # world-fixed gravel dots (give Lucas-Kanade something to track)
        for X, Z, val in dots:
            Zc = Z - cam_z
            if Zc < cam.min_depth:
                continue
            u, v = cam.project(X, Zc)
            u, v = int(u + sx), int(v + sy)
            if 0 <= u < W - 1 and 0 <= v < H - 1:
                s = max(1, int(round(40.0 / Zc)))
                c = (int(val),) * 3
                cv2.rectangle(img, (u, v), (u + s, v + s), c, -1)
        # potholes
        for p in scene.potholes:
            if p.Z - p.rz - cam_z < cam.min_depth:
                continue
            poly = []
            for a in np.linspace(0, 2 * math.pi, 40, endpoint=False):
                u, v = cam.project(p.X + p.rx * math.cos(a), p.Z + p.rz * math.sin(a) - cam_z)
                poly.append((int(round(u + sx)), int(round(v + sy))))
            cv2.fillPoly(img, [np.array(poly, np.int32)], (25, 25, 25))
        # MTWs, far to near
        for mid, b in sorted(ft.mtws.items(), key=lambda kv: kv[1][3]):
            x1, y1, x2, y2 = [int(round(c)) for c in b]
            w, h = x2 - x1, y2 - y1
            if w < 2 or h < 2:
                continue
            cv2.rectangle(img, (x1, y1 + h // 3), (x2, y2 - h // 5), (40, 40, 160), -1)           # body
            cv2.circle(img, ((x1 + x2) // 2, y1 + h // 6), max(1, h // 6), (60, 120, 220), -1)    # rider head
            cv2.circle(img, ((x1 + x2) // 2, y2 - h // 10), max(1, h // 8), (10, 10, 10), -1)     # wheel
        writer.write(img)
    writer.release()


def export_mock(truth: list[FrameTruth], scene: Scene, path: Path, conf: float = 0.9) -> None:
    """Mock detections JSON: the schema read by ptr.detection.detector.MockDetections."""
    frames = []
    for ft in truth:
        frames.append({
            "frame": ft.frame,
            "potholes": [{"bbox_xyxy": [round(c, 2) for c in b], "conf": conf} for b in ft.potholes.values()],
            "mtws": [{"track_id": mid, "bbox_xyxy": [round(c, 2) for c in b], "conf": conf}
                     for mid, b in ft.mtws.items()],
        })
    doc = {"fps": scene.fps, "width": scene.cam.width, "height": scene.cam.height, "source": "synthetic",
           "frames": frames}
    Path(path).write_text(json.dumps(doc))
