# Model card: pothole detector (TEMPLATE, fill in after training; do not invent values)

| field | value |
|---|---|
| file | `models/pothole/best.pt` (never committed to git) |
| SHA-256 | TO FILL (printed by the Colab notebook) |
| date trained | TO FILL |
| base weights | yolov8s.pt (COCO) / TO FILL |
| ultralytics version | TO FILL (tested: 8.4.212) |
| config | `configs/train_pothole.yaml` + run folder `run_config.yaml` |
| dataset version | `data/processed/pothole/manifest.csv` hash TO FILL; sources and licences: docs/ATTRIBUTION.md |
| train / val / Test A images | TO FILL (from `data/processed/pothole/stats.md`) |
| chosen confidence | TO FILL (F1-optimal on VAL, from `eval_results/detection_*.md`) |
| val metrics | TO FILL |
| Test A metrics (held-out source) | TO FILL |
| Test B metrics (annotated video frames) | TO FILL, with image/box counts |
| known failure modes | TO FILL after the failure gallery review |
