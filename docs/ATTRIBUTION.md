# Attribution

Third-party data and videos are never committed to this repository. Every dataset or video you actually use must be listed
here with its URL and the licence text copied from its page. **Entries marked TO VERIFY have not been checked by the author.**

## Software
- Ultralytics YOLOv8 + built-in ByteTrack: https://github.com/ultralytics/ultralytics . Licence: AGPL-3.0 (stated in the project blueprint; TO VERIFY on the repository). Mention it in the report.
- ByteTrack: Zhang, Y. et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022 (cited in the project PPT).
- COCO-pretrained weights (`yolov8s.pt`, downloaded by Ultralytics): COCO dataset, https://cocodataset.org . Licence of annotations/images: TO VERIFY.
- Python libraries: numpy, pandas, PyYAML, OpenCV, SciPy, ImageHash, Pillow, motmetrics, matplotlib, scikit-learn, PyTorch (see requirements*.txt for versions; their licences were not reviewed here).

## Datasets (fill in the licence text you copied)
| dataset | URL | citation to use | licence text (copied) | used for | date accessed |
|---|---|---|---|---|---|
| RDD2022 (India, Czech, China_MotorBike) | https://github.com/sekilab/RoadDamageDetector | Arya et al., RDD2022 (arXiv 2209.08538; TO VERIFY the final reference) | TO VERIFY | training | TO FILL |
| Potholes dataset with YOLO annotations (Bucko et al.) | https://figshare.com/articles/figure/Potholes_dataset_with_YOLO_annotations/21214400 | Sensors 2022, 22, 8878 (as requested on the page; TO VERIFY authors/title) | TO VERIFY | Test A | TO FILL |
| Annotated Water-Filled and Dry Potholes (Dib et al.) | https://data.mendeley.com/datasets/tp95cdvgm8 | TO VERIFY (a data-in-brief article accompanies the record) | TO VERIFY | training | TO FILL |
| Road Surface Irregularities (VILNIUS TECH) | https://zenodo.org/records/18789662 (TO VERIFY) | TO VERIFY | TO VERIFY | optional training | TO FILL |
| Roboflow projects | one row per project URL | per project | per project | optional training | TO FILL |
| IDD (Detection / Multimodal) | https://idd.insaan.iiit.ac.in | TO VERIFY | TO VERIFY | optional MTW evaluation / video | TO FILL |

## Videos
Listed one by one in `data/videos/PROVENANCE.csv` (URL, title, channel, licence shown, access date, time range). The videos are not redistributed.

## This code
No licence file is included (none was requested). Add one if you publish it.
