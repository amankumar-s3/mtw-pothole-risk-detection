import dataclasses
import shutil

import pytest
import yaml

from ptr.config import (ConfigError, PipelineConfig, build, load_pipeline_config, load_risk_config,
                        load_yaml, to_dict)


def test_pipeline_config_loads(pipe_cfg):
    assert pipe_cfg.camera_mode == "moving"
    assert pipe_cfg.detectors.mtw.coco_class_id == 3
    assert pipe_cfg.ego_motion.enabled is True
    assert isinstance(pipe_cfg.ego_motion.roi, list) and len(pipe_cfg.ego_motion.roi) == 4


def test_risk_config_loads_and_normalises(risk_cfg):
    assert "initial hypotheses" in risk_cfg.status
    mov = risk_cfg.for_mode("moving")
    w = mov.weights
    assert abs(w.proximity + w.ttr + w.swerve + w.size + w.decel - 1.0) < 1e-9
    assert w.decel == 0.0                      # deceleration is off in moving mode by default
    sta = risk_cfg.for_mode("static")
    assert sta.weights.decel > 0.0


def test_risk_yaml_is_labelled(root):
    text = (root / "configs" / "risk.yaml").read_text()
    assert "initial hypotheses, to be calibrated" in text


def test_unknown_key_rejected(root, tmp_path):
    d = load_yaml(root / "configs" / "pipeline.yaml")
    d["bogus"] = 1
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="unknown keys"):
        load_pipeline_config(p)


def test_missing_key_rejected(root, tmp_path):
    d = load_yaml(root / "configs" / "pipeline.yaml")
    del d["features"]["window_seconds"]
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="missing required key"):
        load_pipeline_config(p)


def test_wrong_type_rejected(root, tmp_path):
    d = load_yaml(root / "configs" / "pipeline.yaml")
    d["features"]["min_samples"] = "six"
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError):
        load_pipeline_config(p)


def test_invalid_camera_mode(root, tmp_path):
    d = load_yaml(root / "configs" / "pipeline.yaml")
    d["camera_mode"] = "flying"
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError):
        load_pipeline_config(p)


def test_risk_status_must_be_labelled(root, tmp_path):
    d = load_yaml(root / "configs" / "risk.yaml")
    d["status"] = "calibrated, trust me"
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="initial hypotheses"):
        load_risk_config(p)


def test_risk_class_cuts_validated(root, tmp_path):
    d = load_yaml(root / "configs" / "risk.yaml")
    d["moving"]["classes"] = {"low_max": 0.7, "high_min": 0.6}
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError):
        load_risk_config(p)


def test_to_dict_roundtrip(pipe_cfg):
    d = to_dict(pipe_cfg)
    again = build(PipelineConfig, d)
    assert again == pipe_cfg


def test_other_yaml_files_parse(root):
    for name in ["bytetrack_custom.yaml", "dataset_pothole.yaml", "train_pothole.yaml", "data_prep.yaml"]:
        assert isinstance(load_yaml(root / "configs" / name), dict)
    for name in ["rdd2022", "bucko", "dib", "zenodo_vilnius", "roboflow"]:
        d = load_yaml(root / "configs" / "class_maps" / f"{name}.yaml")
        assert d["source"] == name and d["format"] in ("voc", "yolo")


def test_frozen(pipe_cfg):
    with pytest.raises(dataclasses.FrozenInstanceError):
        pipe_cfg.camera_mode = "static"  # type: ignore[misc]
