import json
import re
from pathlib import Path

import pytest
import yaml

import train_pothole

ROOT = Path(__file__).resolve().parents[1]


def test_training_config_disables_vertical_flip_and_rotation():
    cfg = yaml.safe_load((ROOT / "configs" / "train_pothole.yaml").read_text())
    assert cfg["augment"]["flipud"] == 0.0 and cfg["augment"]["degrees"] == 0.0
    assert cfg["model"] == "yolov8s.pt" and cfg["seed"] is not None
    kw = train_pothole.build_args(cfg)
    assert kw["deterministic"] is True and kw["flipud"] == 0.0 and "model" not in kw


def test_dry_run_prints_resolved_arguments(capsys):
    assert train_pothole.main(["--config", str(ROOT / "configs" / "train_pothole.yaml"), "--dry-run", "--imgsz", "960",
                               "--model", "yolov8n.pt", "--name", "abl"]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out["model"] == "yolov8n.pt" and out["train_args"]["imgsz"] == 960 and out["train_args"]["name"] == "abl"


def test_missing_dataset_yaml_is_a_clean_error(tmp_path):
    assert train_pothole.main(["--config", str(ROOT / "configs" / "train_pothole.yaml"), "--data", str(tmp_path / "x.yaml")]) == 2


def test_resume_without_checkpoint_is_a_clean_error(tmp_path):
    pytest.importorskip("ultralytics")
    ds = tmp_path / "d.yaml"
    ds.write_text("path: .\ntrain: a\nval: a\nnames:\n  0: pothole\n")
    assert train_pothole.main(["--config", str(ROOT / "configs" / "train_pothole.yaml"), "--resume", "--data", str(ds),
                               "--project", str(tmp_path / "runs"), "--name", "none"]) == 2


def test_colab_notebook_is_valid_and_follows_the_rules():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(ROOT / "notebooks" / "colab_train_pothole.ipynb", as_version=4)
    nbformat.validate(nb)
    text = "\n".join(c["source"] for c in nb.cells)
    assert "drive.mount" in text
    assert "shutil.copy" in text and "/content/dataset" in text                  # dataset copied to local disk
    assert "--resume" in text and "last.pt" in text                              # resume support
    assert "torch_before == torch_after" in text                                 # Colab's torch is left alone
    assert not re.search(r"pip install[^\n]*\btorch", text)                      # never installs torch
    assert "train_pothole.py" in text and "sha256" in text
    assert "http" not in re.sub(r"https?://colab", "", text)                      # no invented URLs
