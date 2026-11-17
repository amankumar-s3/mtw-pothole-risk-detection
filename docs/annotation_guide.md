# Annotation guide (Test B: potholes + motorcycles, YOLO format)

Goal: 150-300 frames sampled about once per second from at least 8 different public videos (`scripts/extract_frames.py`),
annotated with two classes. Tools: CVAT, Roboflow Annotate or Label Studio (LabelImg is archived: avoid).

## Class ids (do not change)
| id | class | label |
|---|---|---|
| 0 | pothole | a visible depression or broken-out hole in the carriageway |
| 1 | MTW | a motorcycle or scooter (with its rider): label the vehicle as one box |

## Pothole: label
- A visible depression or broken-out hole in the carriageway (asphalt/concrete/dirt road surface the traffic uses).
- Water-filled holes only if the hole edge or depression is clearly visible.
## Pothole: do NOT label
- manhole covers, utility patches/trench repairs, shadows, oil stains, puddles with no visible hole, speed breakers/bumps, road markings, cracks without a hole, potholes on footpaths or far off the carriageway.
- If unsure whether it is a pothole: skip it and write the frame name in the notes file (do not guess). Unsure = not labelled is consistent for both annotators.

## MTW
- One box around the whole motorcycle/scooter plus rider (wheels to head). Pillion passengers are inside the same box.
- Parked motorcycles: label them too if clearly visible (the detector cannot know they are parked).
- Bicycles, e-bikes with pedals only, three-wheelers, autorickshaws, cars: not MTW.

## Boxes, occlusion, truncation
- Tight boxes. Minimum size: the importer rejects boxes smaller than `test_b.min_box_px` (4 px) in either side; do not label objects you cannot recognise.
- Occluded objects: label the visible part only if more than about half of the object is visible; otherwise skip.
- Truncated at the image border: label the visible part (box stays inside the image).

## Pre-labels
`scripts/prelabel.py` writes editable YOLO pre-labels (pothole = 0, MTW = 1). Annotate a first batch from scratch before the
pothole model exists. Always CORRECT pre-labels by hand: delete wrong boxes, add missing ones, fix class ids.

## Export settings
- CVAT: Export annotations > **YOLO 1.1** with images. Class order in the task labels must be `pothole`, `MTW` (ids 0, 1).
- Roboflow: export **YOLOv8** format with classes `pothole`, `MTW` in that order; **disable** all preprocessing/augmentation in the export (no resize, no flips).
- Keep file names `<clip_id>_<frame>.jpg` unchanged.
- Then: `python scripts/import_test_b.py --annotations <export folder> --provenance data/videos/PROVENANCE.csv` (validates class ids in {0, 1}, boxes inside the image, counts distinct source videos).

## Double-annotation agreement (20 frames)
1. Pick 20 frames at random (e.g. `ls | shuf -n 20 --random-source=<(yes)` for a repeatable draw).
2. Two people annotate them independently from scratch (no pre-labels), each exporting YOLO labels to their own folder.
3. `python scripts/check_agreement.py --a labels_A --b labels_B --name test_b_20` (boxes matched at IoU 0.5, per class).
4. Discuss disagreements, update this guide if a rule was unclear, and re-annotate the affected frames. Report the agreement together with the counts it is based on.
