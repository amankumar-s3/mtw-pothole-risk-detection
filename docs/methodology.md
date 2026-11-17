# Methodology

All numbers named here live in `configs/pipeline.yaml` and `configs/risk.yaml`. The risk numbers are **initial hypotheses, to be
calibrated** (not literature values). The output is a potential-risk indicator for a configuration and behaviour pattern in the
video; it is not accident prediction, and not a statement about real speed, real distance or rider intent.

## 1. Camera mode
`camera_mode: moving` (primary; forward-facing dashcam) or `static` (kept as an option). The mode selects the block in `risk.yaml`
and the pothole registry behaviour (static: confirmed potholes may persist for the whole video).

## 2. Ground-plane proxy (analysis/geometry.py)
Assumes a roughly flat road and a forward-facing camera: the image row of a ground contact point increases as the object gets closer.
- MTW contact point = bottom-centre of its (smoothed) box; pothole bottom row = `y2` of its box.
- **dy = (mtw_contact_row - pothole_bottom_row) / mtw_box_height**. dy > 0: MTW closer to the camera than the pothole.
- **dx** = horizontal distance between the MTW contact x and the pothole box (0 if inside) / mtw_box_width.
- **Encounter** = zero crossing of dy. `away`: dy goes + to - (rider moving away from the camera vehicle); `oncoming`: - to +.
  Both directions are tested on synthetic scenes (tests/test_features.py).
- **Clearance** = dx at the crossing (linear interpolation between the two samples); without a crossing, the window minimum, flagged `clearance_from_crossing=false`.
- A pair is **in range** while |dy| <= `features.max_abs_dy` and dx <= `features.max_dx`. Rows stop `features.post_encounter_seconds` after the encounter.

## 3. Features (analysis/features.py), over a window of `features.window_seconds`
All window samples are first brought into the CURRENT frame's coordinates with the ego-motion transforms of that window only.
- **proximity** = minimum over the window of sqrt(dx^2 + dy^2) (units: MTW widths/heights mixed, as in Blueprint Rev 2).
- **TTR** = |dy| / (-slope of |dy| over the window), only while |dy| is shrinking; seconds from video timestamps.
- **swerve** = |least-squares slope of the lateral offset| in MTW widths per second, with offset = (mtw_contact_x - pothole_centre_x) / mtw_width (signed value also logged).
- **decel** = (mean q over the first half of the window - mean q over the second half) / mean q over the first half, with q = |d(contact row)/dt| / box_height^2.
  q is used instead of "speed / box height" because for a ground-plane pinhole camera the image row velocity scales like 1/Z^2 and the box height like 1/Z, so
  q = |dZ/dt| times a constant and is constant for constant-speed motion, whereas speed/height still changes with depth (an MTW riding away at constant speed
  would show a false 'deceleration'). Standstill (mean speed below `features.standstill_speed` box-heights/s) gives NaN and a flag.
  **This is a deliberate deviation from the blueprint text**; it is tested (constant approach gives decel about 0; braking in a static scene gives a large value).
- **size class**: pothole width / MTW width when |dy| <= `features.size.max_abs_dy` (thresholds `ratio_small_max`, `ratio_large_min`), otherwise the pothole box area as a fraction of the frame (thresholds `frac_small_max`, `frac_large_min`).
- Validity flags: `insufficient_history` (fewer than `min_samples` samples in the window), `no_crossing`, `ego_invalid`, `pothole_last_known`, `standstill`.
- **Last-known pothole**: if the registry no longer reports a pothole (e.g. the MTW covers it), the pair keeps the last box for `pothole_memory_seconds`, mapped forward with the ego transforms, and flags it.

### Optional perspective correction (off by default)
An off-axis pothole at a different depth than the MTW makes a straight rider look like he drifts laterally (the pothole's offset from the principal point scales with 1/depth).
`features.perspective.enabled: true` rescales the pothole x to the MTW's depth using a horizon estimate. Measured on the synthetic scenes of tests/test_features.py (maximum swerve in MTW widths/s; potholes 0.5 m or 2.5 m off the camera axis; MTW riding straight or swerving 1 m):

| scene | raw (default) | correct horizon | horizon +0.04 of frame height | horizon -0.04 | horizon +0.08 |
|---|---|---|---|---|---|
| straight, pothole 0.5 m off axis | 0.24 | 0.00 | 1.41 | 0.09 | 0.24 |
| late swerve, pothole 0.5 m off axis | 1.93 | 2.10 | 2.83 | 2.02 | 1.93 |
| straight, pothole 2.5 m off axis | 1.23 | 0.02 | 7.05 | 0.45 | 1.23 |
| late swerve, pothole 2.5 m off axis | 1.23 | 2.11 | 6.53 | 1.71 | 1.23 |

(Synthetic-scene numbers, not real footage.) With the right horizon it removes the artefact; with a wrong horizon it can make things much worse, hence it is opt-in and guarded by `max_depth_ratio`.

## 4. Ego-motion (analysis/egomotion.py)
Lucas-Kanade on `goodFeaturesToTrack` corners inside an ROI band and OUTSIDE all detected boxes (grown by a margin), RANSAC `estimateAffinePartial2D`.
Valid if enough points and `inlier_ratio >= min_inlier_ratio`; otherwise identity with `valid=false` and the reason in `egomotion.csv`. With `use_scale: false`
(default) only translation + rotation are kept (forward-motion expansion depends on depth and would distort MTW boxes). Transforms are composed only inside
windows (`max_span_seconds`), never accumulated over the video. The relative features work without it; it helps shaky footage and last-known potholes.

## 5. Pothole registry (analysis/pothole_registry.py)
Greedy IoU association of detections to predicted boxes (running median + residual velocity, mapped by ego-motion), confirmation after `min_hits`, drop after `max_age_seconds`,
median of the last `median_window` boxes (expressed in a common frame via ego-motion), merge of entries overlapping more than `merge_iou`. IDs are short-term in moving mode.
(Blueprint Rev 2 mentions "Ultralytics track with ByteTrack" for potholes in one place and "IoU plus motion prediction" in another; the latter is implemented, ByteTrack is used for MTWs only.)

## 6. Risk (analysis/risk.py, configs/risk.yaml)
Each indicator is a linear map to [0, 1] (0 at `zero_at`, 1 at `one_at`, clipped; NaN = unavailable = 0):
proximity P (4 to 1), TTR T (3 s to 1 s), swerve S (0.2 to 1.0 widths/s), decel D (10% to 40%), size Z (small 0.3, medium 0.6, large 1.0).
Moving block weights: proximity 0.30, TTR 0.15, swerve 0.30, size 0.15, decel 0.00 (renormalised to sum 1 at load: 0.333, 0.167, 0.333, 0.167, 0).
Static block: 0.30, 0.15, 0.25, 0.15 (size), 0.15 (decel). Score = sum of weight x indicator.
Classes: Low < 0.35; Moderate 0.35 to < 0.65; High >= 0.65 AND P >= 0.5 AND S >= 0.5 (or D >= 0.5 only if the decel weight is > 0) AND Z >= 0.6.
Each assessment carries a readable reason ("close pass (distance 0.8 MTW widths), strong lateral shift 1.2 widths/s, large pothole; score 0.91").
Blueprint Rev 2 gives the decel weight three different ways (0 in two places, 0.10 in one); the default here is 0 in moving mode.

## 7. Persistence, hysteresis, events (analysis/events.py)
A higher class is accepted after it persisted `accept_seconds` (the lowest class that persisted is accepted); the class drops after the lower class persisted `deescalate_seconds`.
One event per (mtw_id, pothole_id) encounter; it closes after `event_close_seconds` without rows; encounters shorter than `min_event_frames` are not reported
(their number is in `summary.json`). Event peak class = highest accepted class; peak score = highest score while at that class.

## 8. Data and splits
Leave-source-out: Test A = one whole source (default `bucko`) never used in training or validation. Train/val: remaining sources split about 85/15 per source **by perceptual-hash cluster** (clusters are computed jointly over all sources; near-duplicates of Test A images are removed from train/val).
Test B = annotated frames from public videos, at least 8 distinct videos, all frames of a clip in the same set. Cleaning drops corrupt images and boxes smaller than `clean.min_box_px`.
Blueprint v1 said "remove near-duplicates"; Rev 2 says "cluster": clusters are kept so splits can respect them (`--drop-duplicates` removes within-source duplicates).

## 9. Evaluation protocol
Detection: P, R, F1, AP@0.5, AP@0.5:0.95 computed by `ptr.eval.detection` (all-point AP, greedy matching, so numbers can differ slightly from Ultralytics `val`; do not mix them).
The operating confidence is the middle of the F1 plateau on **val only**; Test A and Test B are reported separately with image and box counts, and their gap is reported, not hidden.
Tracking: motmetrics (ID switches, IDF1, MOTA, MOTP), IoU matching computed in `ptr.eval.tracking` (motmetrics 1.4.0's own `iou_matrix` fails with NumPy 2).
Risk: blind double labelling, system-vs-human confusion matrix, accuracy, linearly weighted Cohen's kappa, inter-rater kappa, always with the sample size;
calibration clips and held-out clips are split BY CLIP; sensitivity analysis perturbs each parameter by +/-20%. Fewer than about 30 events: report as an exploratory case study.
Detection/tracking accuracy and risk-label agreement are never conflated.
