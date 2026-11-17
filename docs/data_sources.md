# Data sources (every licence and count is TO VERIFY until you copy it from the source page)

Nothing here was downloaded by the author of this repository. URLs below come from Blueprint Revision 2 and from
page checks made while writing the code; each must be re-checked on the day you download. Licence text must be
copied from each page into the "licence" column. Counts must come from `scripts/verify_datasets.py`, not from this file.

| dataset | source URL | access | licence (copy from page) | format / classes (to confirm) | images (from verify script) | decision | notes |
|---|---|---|---|---|---|---|---|
| RDD2022 India, Czech, China_MotorBike | https://github.com/sekilab/RoadDamageDetector (links to the zips; a Figshare record is also listed there, DOI 10.6084/m9.figshare.21431547) | direct download | TO VERIFY. Blueprint Rev 2 reports a CC BY-SA 4.0 notice on GitHub / in the data paper and a CC BY 4.0 tag on Figshare: follow the stricter one until checked | Pascal VOC XML; pothole = D40, cracks D00/D10/D20 dropped (class map `configs/class_maps/rdd2022.yaml`, TO VERIFY by visual audit of ~100 D40 boxes). Test folders have NO labels | TO FILL | use (train) | the whole zips for Norway (9.9 GB), United_States (Street View) and China_Drone (aerial) are skipped on purpose |
| Bucko et al., "Potholes dataset with YOLO annotations" | https://figshare.com/articles/figure/Potholes_dataset_with_YOLO_annotations/21214400 | direct download | TO VERIFY (the page asks to cite Sensors 2022, 22, 8878; no licence text was seen) | YOLO txt, single class (id to confirm) ; dashcam, Full HD | TO FILL | use as **Test A** (never in training) | near-duplicate frames from a few drives are likely: clustering handles it |
| Dib et al., "Annotated Water-Filled and Dry Potholes" | https://data.mendeley.com/datasets/tp95cdvgm8 (DOI 10.17632/tp95cdvgm8.1) | free download | TO VERIFY (described as Creative Commons; version not confirmed) | VOC XML and YOLO txt both provided: we read the VOC XML only. XML class label text TO VERIFY (`configs/class_maps/dib.yaml`). Two dashcam videos included (faces/plates blurred) | TO FILL | use (train) | UK phone images; videos useful for qualitative checks only |
| Road Surface Irregularities (VILNIUS TECH) | https://zenodo.org/records/18789662 (record number TO VERIFY) | direct download, ~13 GB single zip: use Colab | TO VERIFY | YOLO txt; 3 classes (pothole, manhole, bump); the class id order is NOT confirmed (`configs/class_maps/zenodo_vilnius.yaml`) | TO FILL | optional | manholes and bumps become background |
| Roboflow Universe pothole projects | https://universe.roboflow.com (record each project URL) | free account, YOLO export | per project, TO VERIFY | YOLO txt; class ids differ per project (`configs/class_maps/roboflow.yaml`) | TO FILL | optional supplement | often contain augmented duplicates and their own train/valid overlaps: we re-split ourselves |
| Kaggle pothole sets named in the proposal (chitholian, anggadwisunarto) | https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset , https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8 | Kaggle login | TO VERIFY (the proposal claims CC0 for one: not verified) | TO VERIFY | TO FILL | optional, audit first | not checked by the author |
| IDD Detection (IIIT Hyderabad) | https://idd.insaan.iiit.ac.in | free registration, approval time unknown | TO VERIFY | VOC XML (to confirm); class names to confirm after download | TO FILL | optional: evaluates the COCO motorcycle detector on Indian roads | convert motorcycle -> class 0 yourself; do not quote sizes until checked |
| IDD Multimodal | https://idd.insaan.iiit.ac.in | free registration | TO VERIFY | front-camera image sequences (15 fps per Rev 2): stitch into video | n/a | optional video source (MTW-rich) | potholes are not labelled |
| Public dashcam / ride videos (e.g. YouTube) | you choose and record in `data/videos/PROVENANCE.csv` | per video | per video, copyright and platform terms apply | n/a | n/a | main source of pothole + MTW footage for Test B and pipeline tests | rules below |

## Not used
- BDD100K (not India, motorcycles very rare in its tracking labels, non-commercial terms).
- A commercial listing claiming "1,000 hours of road video" that could not be verified.
- BharatPotHole / iWatchRoad: a public download could not be confirmed.
- IEEE DataPort "Road Lane and Traffic Density India": needs an institutional subscription (optional if your library has it).
- Two further recent pothole datasets surfaced by a web search while writing this (a Bridgestone urban pothole set and an Iraqi road-defect set): **not evaluated**, only leads. Check licences and class definitions before any use.

## Public video curation protocol (for the Test B and pipeline-evaluation clips)
- 12-20 clips of 30-90 s from at least 6 different videos; Test B needs frames from **at least 8 distinct videos**.
- Accept only: forward-facing dashcam/windshield view, daylight, 720p or better, visible potholes AND MTWs ahead, not extremely shaky.
- Log every clip in `data/videos/PROVENANCE.csv` (columns: clip_id, source, url, title, channel, licence_shown, date_accessed, start_s, end_s, camera_view, daylight, mtw_visible, potholes_visible, notes).
- Raw videos stay OUT of git and OUT of the submitted package (`data/videos/*` is git-ignored except PROVENANCE.csv): submit URLs and timestamps.
- Mask faces and number plates in any screenshot you publish. Check your institution's policy on third-party video. This is not legal advice.
- Name frames `<clip_id>_<frame>.jpg` (`scripts/extract_frames.py` does this) so that a clip is never split across sets.
