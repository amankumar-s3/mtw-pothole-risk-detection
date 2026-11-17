# Tracking ground truth (MOT format) for MTW

Goal: 2-3 short clips (20-30 s) with one persistent ID per motorcycle, used by `scripts/evaluate_tracking.py` to compute
ID switches, IDF1, MOTA and MOTP. Annotate in CVAT.

## Rules
- Class: motorcycle/scooter with rider ONLY (same definition as docs/annotation_guide.md). No potholes in the MOT files.
- One ID per rider for the whole clip. If a rider is hidden by another vehicle and re-appears, it keeps the SAME ID (this is what makes ID switches measurable).
- Box every frame the vehicle is visible (CVAT interpolation between keyframes is fine; check it at turns and occlusions).
- Visible part only when occluded more than about half: skip frames where less than half is visible.
- The clip must be cut from a video listed in `data/videos/PROVENANCE.csv`; use the same file you process with the pipeline (same frames, same fps).

## Export
- CVAT: Export task dataset > **MOT 1.1**. Take `gt/gt.txt`.
- Save as `data/mot_gt/<clip_id>/gt.txt`. Lines are `frame,id,x,y,w,h,conf,class,visibility` with 1-based frame numbers; only the first six columns are used.

## Scoring
`python scripts/run_pipeline.py --video data/videos/<clip>.mp4 --out outputs`
`python scripts/evaluate_tracking.py --gt data/mot_gt/<clip_id>/gt.txt --run outputs/<run_id>`
Frame mapping: gt_frame = processed_frame * frame_stride + 1; only frames the pipeline processed are scored. The IoU threshold
is `tracking_eval.iou_threshold` in `configs/data_prep.yaml`. State the clip length and number of tracks next to every number:
with 2-3 short clips these metrics are indicative, not statistical.
