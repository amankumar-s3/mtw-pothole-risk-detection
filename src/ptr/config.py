"""Typed, strict YAML configuration loading.

There are deliberately NO default values on the config dataclasses: every number must come from a YAML
file (configs/pipeline.yaml, configs/risk.yaml). A missing or unknown key is an error.
"""
from __future__ import annotations

import dataclasses
import math
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar

import yaml

T = TypeVar("T")

CAMERA_MODES = ("moving", "static")


class ConfigError(ValueError):
    """Raised for any invalid configuration."""


# --------------------------------------------------------------------------- generic builder
def _convert(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return build(tp, value, path)
    if origin is typing.Union or origin is getattr(types, "UnionType", object()):
        if value is None and type(None) in args:
            return None
        last: Optional[ConfigError] = None
        for a in args:
            if a is type(None):
                continue
            try:
                return _convert(a, value, path)
            except ConfigError as exc:
                last = exc
        raise last or ConfigError(f"{path}: invalid value {value!r}")
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{path}: {value!r} not in {list(args)}")
        return value
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        inner = args[0] if args else Any
        out = [_convert(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(out) if origin is tuple else out
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        kt, vt = args if args else (Any, Any)
        return {_convert(kt, k, path): _convert(vt, v, f"{path}.{k}") for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported config type {tp!r}")


def build(cls: type[T], data: Any, path: str = "") -> T:
    """Recursively build dataclass ``cls`` from a mapping; strict about missing/unknown keys."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        sub = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], sub)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{sub}: missing required key")
    return cls(**kwargs)


def to_dict(obj: Any) -> Any:
    """Dataclass tree -> plain dict (for config_used.yaml)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------- pipeline config
@dataclass(frozen=True)
class VideoCfg:
    fallback_fps: float
    timestamp_mode: Literal["index", "msec"]
    frame_stride: int
    max_frames: Optional[int]
    resize_width: Optional[int]
    writer_fourcc: str


@dataclass(frozen=True)
class PotholeDetCfg:
    weights: str
    conf: float
    iou: float
    imgsz: int
    max_det: int
    min_box_px: float
    device: Optional[str]


@dataclass(frozen=True)
class MtwDetCfg:
    weights: str
    coco_class_id: int
    conf: float
    iou: float
    imgsz: int
    min_box_px: float
    device: Optional[str]
    tracker_config: str


@dataclass(frozen=True)
class DetectorsCfg:
    pothole: PotholeDetCfg
    mtw: MtwDetCfg


@dataclass(frozen=True)
class TrajectoryCfg:
    ema_alpha: float
    min_history_frames: int
    max_missed_frames: int
    min_reliable_seconds: float
    velocity_frames: int


@dataclass(frozen=True)
class RegistryCfg:
    iou_match: float
    min_hits: int
    max_age_seconds: float
    tentative_max_age_seconds: float
    median_window: int
    velocity_alpha: float
    merge_iou: float
    static_persistent: bool


@dataclass(frozen=True)
class EgoMotionCfg:
    enabled: bool
    roi: list[float]
    max_corners: int
    quality_level: float
    min_distance: float
    block_size: int
    box_margin_px: int
    lk_win: int
    lk_levels: int
    ransac_thresh_px: float
    min_points: int
    min_inlier_ratio: float
    use_scale: bool
    max_span_seconds: float


@dataclass(frozen=True)
class SizeCfg:
    max_abs_dy: float
    ratio_small_max: float
    ratio_large_min: float
    frac_small_max: float
    frac_large_min: float


@dataclass(frozen=True)
class RegistrySizeCfg:
    frac_small_max: float
    frac_large_min: float


@dataclass(frozen=True)
class PerspectiveCfg:
    enabled: bool
    horizon_row_frac: float      # horizon row as a fraction of the frame height (flat-road assumption)
    principal_x_frac: float
    max_depth_ratio: float       # if the implied depth ratio is outside [1/r, r] the raw offset is used


@dataclass(frozen=True)
class FeaturesCfg:
    window_seconds: float
    min_samples: int
    max_abs_dy: float
    max_dx: float
    pothole_memory_seconds: float
    pair_expire_seconds: float
    min_slope_eps: float
    standstill_speed: float
    ego_compensation: bool
    post_encounter_seconds: float
    perspective: PerspectiveCfg
    size: SizeCfg
    registry_size: RegistrySizeCfg


@dataclass(frozen=True)
class OutputCfg:
    trail_length: int
    font_scale: float
    draw_ego_indicator: bool
    write_video: bool


@dataclass(frozen=True)
class PipelineConfig:
    camera_mode: Literal["moving", "static"]
    video: VideoCfg
    detectors: DetectorsCfg
    trajectory: TrajectoryCfg
    registry: RegistryCfg
    ego_motion: EgoMotionCfg
    features: FeaturesCfg
    output: OutputCfg


def validate_pipeline(cfg: PipelineConfig) -> None:
    if cfg.video.frame_stride < 1:
        raise ConfigError("video.frame_stride must be >= 1")
    if not 0.0 < cfg.trajectory.ema_alpha <= 1.0:
        raise ConfigError("trajectory.ema_alpha must be in (0, 1]")
    if cfg.trajectory.min_history_frames < 2:
        raise ConfigError("trajectory.min_history_frames must be >= 2")
    if len(cfg.ego_motion.roi) != 4:
        raise ConfigError("ego_motion.roi must have 4 numbers")
    x0, y0, x1, y1 = cfg.ego_motion.roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ConfigError("ego_motion.roi must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    if not 0.0 < cfg.ego_motion.min_inlier_ratio <= 1.0:
        raise ConfigError("ego_motion.min_inlier_ratio must be in (0, 1]")
    if cfg.features.min_samples < 3:
        raise ConfigError("features.min_samples must be >= 3")
    if cfg.features.window_seconds <= 0:
        raise ConfigError("features.window_seconds must be > 0")
    if cfg.features.post_encounter_seconds < 0:
        raise ConfigError("features.post_encounter_seconds must be >= 0")
    if cfg.features.perspective.max_depth_ratio <= 1.0:
        raise ConfigError("features.perspective.max_depth_ratio must be > 1")
    if not 0.0 <= cfg.features.perspective.horizon_row_frac < 1.0:
        raise ConfigError("features.perspective.horizon_row_frac must be in [0, 1)")
    if cfg.registry.min_hits < 1:
        raise ConfigError("registry.min_hits must be >= 1")
    if cfg.features.size.ratio_small_max >= cfg.features.size.ratio_large_min:
        raise ConfigError("features.size.ratio_small_max must be < ratio_large_min")
    if cfg.features.size.frac_small_max >= cfg.features.size.frac_large_min:
        raise ConfigError("features.size.frac_small_max must be < frac_large_min")


# --------------------------------------------------------------------------- risk config
@dataclass(frozen=True)
class Weights:
    proximity: float
    ttr: float
    swerve: float
    size: float
    decel: float


@dataclass(frozen=True)
class Anchor:
    zero_at: float
    one_at: float


@dataclass(frozen=True)
class Anchors:
    proximity: Anchor
    ttr: Anchor
    swerve: Anchor
    decel: Anchor


@dataclass(frozen=True)
class SizeScores:
    small: float
    medium: float
    large: float


@dataclass(frozen=True)
class ClassCuts:
    low_max: float
    high_min: float


@dataclass(frozen=True)
class HighGate:
    p_min: float
    s_min: float
    b_min: float
    z_min: float


@dataclass(frozen=True)
class Persistence:
    accept_seconds: float
    deescalate_seconds: float
    event_close_seconds: float
    min_event_frames: int


@dataclass(frozen=True)
class RiskModeCfg:
    weights: Weights
    anchors: Anchors
    size_scores: SizeScores
    classes: ClassCuts
    high_gate: HighGate
    persistence: Persistence


@dataclass(frozen=True)
class RiskConfig:
    status: str
    normalize_weights: bool
    moving: RiskModeCfg
    static: RiskModeCfg

    def for_mode(self, camera_mode: str) -> RiskModeCfg:
        """Select the block for the camera mode (weights normalised if requested)."""
        if camera_mode not in CAMERA_MODES:
            raise ConfigError(f"camera_mode must be one of {CAMERA_MODES}, got {camera_mode!r}")
        block = getattr(self, camera_mode)
        if not self.normalize_weights:
            return block
        w = block.weights
        total = w.proximity + w.ttr + w.swerve + w.size + w.decel
        nw = Weights(w.proximity / total, w.ttr / total, w.swerve / total, w.size / total, w.decel / total)
        return dataclasses.replace(block, weights=nw)


def _validate_risk_block(name: str, b: RiskModeCfg) -> None:
    w = b.weights
    vals = [w.proximity, w.ttr, w.swerve, w.size, w.decel]
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ConfigError(f"{name}.weights must be >= 0 with a positive sum")
    for aname in ("proximity", "ttr", "swerve", "decel"):
        a: Anchor = getattr(b.anchors, aname)
        if a.zero_at == a.one_at:
            raise ConfigError(f"{name}.anchors.{aname}: zero_at must differ from one_at")
    ss = b.size_scores
    if not all(0 <= v <= 1 for v in (ss.small, ss.medium, ss.large)):
        raise ConfigError(f"{name}.size_scores must be in [0, 1]")
    if not 0 < b.classes.low_max < b.classes.high_min <= 1:
        raise ConfigError(f"{name}.classes needs 0 < low_max < high_min <= 1")
    p = b.persistence
    if p.accept_seconds < 0 or p.deescalate_seconds < 0 or p.event_close_seconds <= 0 or p.min_event_frames < 1:
        raise ConfigError(f"{name}.persistence has invalid values")


def validate_risk(cfg: RiskConfig) -> None:
    if "initial hypotheses" not in cfg.status.lower() or "calibrated" not in cfg.status.lower():
        raise ConfigError('risk.yaml status must say "initial hypotheses, to be calibrated"')
    _validate_risk_block("moving", cfg.moving)
    _validate_risk_block("static", cfg.static)


# --------------------------------------------------------------------------- loaders
def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return data


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    cfg = build(PipelineConfig, load_yaml(path))
    validate_pipeline(cfg)
    return cfg


def load_risk_config(path: str | Path) -> RiskConfig:
    cfg = build(RiskConfig, load_yaml(path))
    validate_risk(cfg)
    return cfg


def resolve_path(p: str, base: Optional[Path] = None) -> str:
    """Resolve a config path against ``base`` (default: cwd) if it exists there; otherwise return as is
    (so bare names such as ``yolov8s.pt`` still trigger the Ultralytics auto-download)."""
    q = Path(p)
    if q.is_absolute():
        return str(q)
    cand = (base or Path.cwd()) / q
    return str(cand) if cand.exists() else p


def finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
