# Pothole and risk detection for motorized two-wheeler (MTW) users

Computer-vision pipeline that finds potholes and MTWs in **offline road video**, tracks the MTWs, measures how each MTW passes
each pothole and reports an explainable **Low / Moderate / High potential-risk** label with a readable reason for every encounter.
It is a potential-risk indicator, **not accident prediction**.

Authors / supervisor / university: _(fill in)_. Code licence: none included.

In scope: potholes + MTWs, offline video analysis, YOLOv8 + ByteTrack, rule-based explainable risk. Out of scope: GPS, backend,
dashboard, alerts, edge hardware, cyclists, pedestrians.

> **Status.** Everything below that needs real data (datasets, trained weights, real footage, GPU training) has to be done by you.
> The code, the tests, the mock mode and a tiny CPU training run were exercised by the author (see "Tested" at the end);
> no accuracy number for real roads exists yet and none is claimed anywhere in this repository.

Design in one paragraph: two separate detectors (A: single-class pothole YOLOv8s fine-tuned from COCO; B: COCO YOLOv8s restricted to
class 3 "motorcycle" + Ultralytics ByteTrack), primary camera mode = moving dashcam, ground-plane proxy features that are relative between MTW
and pothole (depth gap `dy`, lateral gap `dx`, encounter = zero crossing of `dy`, clearance, proximity, time-to-reach, swerve, size class; deceleration is logged but has
weight 0 in moving mode), optional ego-motion compensation, weighted risk score with a gated High class, persistence + hysteresis. Details: `docs/methodology.md`,
`docs/architecture.md`, `docs/limitations.md`.

---------------------------------------------------------------------------------------------------------------------------------

## 0. Install (Ubuntu)

Run everything from the repository root.
```bash
python3 -m venv .venv && source .venv/bin/activate        # tested with Python 3.12.3 (3.11 was not available to the author: untested)
pip install -r requirements.txt
pip install -e .
pytest                                                     # no GPU, no downloads (about one minute on a 1-core sandbox)
```
Real detectors (needed for training, prelabel, detector evaluation and real videos; NOT needed for tests or mock mode):
```bash
pip install --no-deps ultralytics==8.4.212
pip install -r requirements-detect.txt                      # torch, torchvision, lap (needed by ByteTrack), ...
```

## 1. Try the whole pipeline without any data or weights (mock mode)
```bash
python scripts/make_synthetic_video.py --out data/videos/synthetic_demo
python scripts/run_pipeline.py --video data/videos/synthetic_demo.mp4 --mock-detections data/videos/synthetic_demo.json --run-id demo
python scripts/make_figures.py --run outputs/demo
```
The synthetic video has a moving, shaking camera, three potholes and three MTWs (one passes wide, one rides through its pothole, one swerves late).
Open `outputs/demo/annotated.mp4`, `events.csv`, `summary.json`. (Browsers/players that cannot play `mp4v`:
`ffmpeg -i outputs/demo/annotated.mp4 -c:v libx264 -pix_fmt yuv420p outputs/demo/annotated_h264.mp4`.)

## 2. Datasets (you do the downloads) and the pothole training set
Which datasets and where: **docs/data_sources.md** (RDD2022 India/Czech/China_MotorBike for training, Bucko dashcam potholes as the held-out Test A, Dib water-filled potholes for
training; optional extras). Licence text of each one is TO VERIFY: copy it from the page into `docs/data_sources.md` and `docs/ATTRIBUTION.md`.
Unzip so that the layout is:
```
data/raw/rdd2022_india/  data/raw/rdd2022_czech/  data/raw/rdd2022_china_motorbike/  data/raw/bucko/  data/raw/dib/
```
Audit each source (counts, class histogram, box sizes) and look at 50 random annotated images:
```bash
python scripts/verify_datasets.py --root data/raw/rdd2022_india --format voc  --name rdd2022_india
python scripts/verify_datasets.py --root data/raw/bucko         --format yolo --name bucko
python scripts/verify_datasets.py --root data/raw/dib           --format voc  --name dib
python scripts/preview_annotations.py --root data/raw/rdd2022_india --format voc --only-label D40 --n 50 --out docs/audits/rdd2022_india_preview.png
python scripts/preview_annotations.py --root data/raw/bucko --format yolo --n 50 --out docs/audits/bucko_preview.png
python scripts/preview_annotations.py --root data/raw/dib   --format voc  --n 50 --out docs/audits/dib_preview.png
```
Check the class histograms against `configs/class_maps/*.yaml` (RDD2022 `D40` = pothole; the Dib XML label text and the Bucko class id are guesses marked TO VERIFY: edit the maps to match
what the histograms show). Then convert, clean, cluster near-duplicates and split (leave-source-out, Bucko = Test A):
```bash
python scripts/convert_to_yolo.py --source rdd2022_india           --input data/raw/rdd2022_india           --class-map configs/class_maps/rdd2022.yaml
python scripts/convert_to_yolo.py --source rdd2022_czech           --input data/raw/rdd2022_czech           --class-map configs/class_maps/rdd2022.yaml
python scripts/convert_to_yolo.py --source rdd2022_china_motorbike --input data/raw/rdd2022_china_motorbike --class-map configs/class_maps/rdd2022.yaml
python scripts/convert_to_yolo.py --source dib                     --input data/raw/dib                     --class-map configs/class_maps/dib.yaml
python scripts/convert_to_yolo.py --source bucko                   --input data/raw/bucko                   --class-map configs/class_maps/bucko.yaml
python scripts/clean_dedupe.py
python scripts/make_splits.py --test-a-source bucko
cat data/processed/pothole/stats.md                         # counts per source and split
(cd data/processed && zip -r pothole_dataset.zip pothole)   # for Colab
```
Optional sources (Zenodo Vilnius, Roboflow projects): same `convert_to_yolo.py` call with their class map after fixing the class ids.
`make_splits.py` fails with exit code 3 if any hash cluster would span two splits; it also removes near-duplicates of Test A images from train/val.
RDD2022 test folders have no labels and are skipped automatically.

## 3. Train the pothole detector on Google Colab
Upload `pothole-mtw-risk.zip` (this project) and `pothole_dataset.zip` to `MyDrive/ptr/`, open `notebooks/colab_train_pothole.ipynb` on a GPU runtime and run it top to bottom
(it mounts Drive, copies the data to local disk, installs only `ultralytics` without touching Colab's torch, trains straight into Drive so `last.pt` survives disconnects, and resumes
automatically when you re-run the training cell). Without Colab (any machine with a GPU), the same step is:
```bash
python scripts/train_pothole.py --config configs/train_pothole.yaml --name yolov8s_640
python scripts/train_pothole.py --config configs/train_pothole.yaml --resume --name yolov8s_640      # after an interruption
python scripts/train_pothole.py --config configs/train_pothole.yaml --model yolov8n.pt --name yolov8n_640   # ablation: speed
python scripts/train_pothole.py --config configs/train_pothole.yaml --imgsz 960 --batch 8 --name yolov8s_960  # ablation: small objects
```
Record every run in `docs/training_log_template.md`. Download `best.pt`, then:
```bash
cp <downloaded>/yolov8s_640_best.pt models/pothole/best.pt && sha256sum models/pothole/best.pt     # fill models/pothole/model_card_template.md
```

## 4. Test B: annotated frames from public videos (>= 8 distinct videos)
Choose clips following the protocol in `docs/data_sources.md`; log each in `data/videos/PROVENANCE.csv`; keep the videos out of git.
```bash
python scripts/extract_frames.py --video data/videos/clip01.mp4 --clip-id clip01 --start 5 --end 60       # ~1 frame/s, blurry frames skipped
python scripts/prelabel.py --images data/interim/test_b_candidates/clip01 --out data/interim/prelabels/clip01 \
       --pothole-weights models/pothole/best.pt --mtw-weights yolov8s.pt                                    # after training; annotate the first batch from scratch
```
Annotate in CVAT/Roboflow following `docs/annotation_guide.md` (class 0 pothole, 1 MTW; export YOLO). Double-annotate 20 frames and check agreement, then import:
```bash
python scripts/check_agreement.py --a labels_annotatorA --b labels_annotatorB --name test_b_20
python scripts/import_test_b.py --annotations exports/cvat_yolo --provenance data/videos/PROVENANCE.csv --check-leak data/processed/pothole
```
It validates class ids/boxes, keeps every clip whole, writes `data/processed/test_b/clips.csv` and prints the REAL number of distinct source videos (warns if < 8; `--strict` fails).
Tracking ground truth (2-3 clips of 20-30 s): `docs/mot_annotation_guide.md` -> `data/mot_gt/<clip_id>/gt.txt`.

## 5. Evaluate the detectors (confidence chosen on val only)
```bash
python scripts/evaluate_detection.py --name pothole_s640 --weights models/pothole/best.pt --target pothole \
       --split val=data/processed/pothole/val --split test_a=data/processed/pothole/test_a --split test_b=data/processed/test_b \
       --gt-class 0 --conf-from val --gallery-split test_b
python scripts/evaluate_detection.py --name mtw_coco --weights yolov8s.pt --target mtw \
       --split test_b=data/processed/test_b --gt-class 1 --conf 0.30
```
(Optional: a YOLO-format folder converted from IDD motorcycle labels, motorcycle = class 0, works the same with `--gt-class 0`.)
Output in `eval_results/`: `detection_<name>.csv|md`, PR/F1 curves, failure gallery. The report states image and box counts, the Test A vs Test B gap and the chosen confidence.
Speed (informational, hardware logged): `python scripts/benchmark_fps.py --video data/videos/clip01.mp4 --pothole-weights models/pothole/best.pt --mtw-weights yolov8s.pt`

## 6. Run the pipeline on real videos
```bash
python scripts/run_pipeline.py --video data/videos/clip01.mp4 --run-id clip01
python scripts/make_figures.py --run outputs/clip01
```
Useful flags: `--camera-mode static`, `--max-frames N`, `--no-video`, `--config`, `--risk-config`. Each run folder holds `annotated.mp4, tracks.csv, potholes.csv, interactions.csv,
events.csv, summary.json, egomotion.csv, config_used.yaml, run.log` (columns documented in `docs/architecture.md`). Tune `configs/bytetrack_custom.yaml` and `configs/pipeline.yaml` on your clips.
The MTW detector downloads `yolov8s.pt` on first use (needs internet).

## 7. System evaluation on the public clips
```bash
python scripts/evaluate_tracking.py --gt data/mot_gt/clip01/gt.txt --run outputs/clip01                      # ID switches, IDF1, MOTA, MOTP
python scripts/export_encounters_for_labeling.py --run outputs/clip01 --run outputs/clip02 --out labeling    # blind clips + empty labels CSV
```
Two raters fill `rater1` / `rater2` in `labeling/labels_template.csv` (Low / Moderate / High, WITHOUT seeing `events.csv`); resolve disagreements in a `consensus` column; save as `labeling/labels.csv`.
```bash
python scripts/split_events_by_clip.py --events outputs/clip01/events.csv outputs/clip02/events.csv --out eval_results   # calibration vs held-out BY CLIP
# calibrate configs/risk.yaml on the calibration clips only, re-scoring without re-running video (event ids stay valid):
python scripts/rescore_run.py --run outputs/clip01 --risk-config configs/risk.yaml --out outputs/clip01/events_rescored.csv
python scripts/evaluate_risk_agreement.py --events eval_results/events_heldout.csv --labels labeling/labels.csv --name heldout
python scripts/sensitivity_analysis.py --run outputs/clip01 --run outputs/clip02 --name sensitivity          # +/-20% on every parameter
```
Always report the number of clips, events and raters with every agreement number. With fewer than about 30 events, present a case study (`docs/limitations.md`).
After calibration, run the final held-out evaluation on events produced with the calibrated `risk.yaml` (`rescore_run.py` writes them; point `--events` at the rescored CSV).

## Folder layout
`configs/` YAML (pipeline, risk, bytetrack, dataset, training, data prep, class maps) · `data/` (git-ignored: raw, interim, processed, mot_gt, videos) ·
`models/` (weights are git-ignored) · `notebooks/` Colab training · `scripts/` command line tools · `src/ptr/` the package · `tests/` pytest ·
`outputs/` run folders · `eval_results/` evaluation reports · `docs/` data sources, annotation guides, architecture (Mermaid), methodology, limitations, attribution.

## Tested (by the author, in a 1-core Ubuntu sandbox, Python 3.12.3)
- `pytest`: see the final report that came with this ZIP for the exact summary. The suite uses synthetic data only.
- Mock end-to-end run (synthetic video + mock detections) inside pytest and from the command line.
- With real packages (ultralytics 8.4.212, torch 2.15.0 CPU): download of COCO `yolov8n.pt`, ByteTrack `model.track` with `configs/bytetrack_custom.yaml`, a tiny 3-epoch training on a toy synthetic dataset,
  a genuine kill-and-resume (checkpoint epoch 10 resumed to 40), `evaluate_detection.py`, `prelabel.py`, and `run_pipeline.py` with the real detectors on 45 frames of the synthetic video.
  Those runs prove the plumbing; their detections and metrics mean nothing about real potholes.
- **Not tested**: real datasets, real trained weights, GPU/Colab training, real road footage, Python 3.11, any accuracy on real data.

## Troubleshooting
- `ModuleNotFoundError: lap` when tracking: `pip install lap==0.5.14` (it is in `requirements-detect.txt`).
- `np.asfarray` error from motmetrics: not an issue here (IoU distances are computed in `ptr/eval/tracking.py`); do not replace that code with `mm.distances.iou_matrix` on NumPy 2.
- `pothole weights not found`: train/copy `models/pothole/best.pt`, or use `--mock-detections`.
- A resume request on a finished run prints "nothing to resume": start a new run name.
- Out of memory while training: lower `--batch` or `--imgsz`.
