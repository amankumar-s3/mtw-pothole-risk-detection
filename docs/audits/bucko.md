# Audit: bucko

Measured by `scripts/verify_datasets.py` (no values were assumed). Licence, source URL and the visual audit must still be filled in by hand in docs/data_sources.md.

- root: `data/raw/bucko`
- format: yolo
- images: 2099
- annotation files: 2099
- boxes: 3608
- images without annotation: 0
- annotations without image: 0
- duplicate image file stems (ignored): 0
- corrupt images: 0

## Class histogram

| class | boxes |
|---|---|
| 0 | 3166 |
| 1 | 442 |

## Relative box size (fraction of image)

- width: {'p5': 0.02187, 'p25': 0.03359, 'p50': 0.04609, 'p75': 0.06328, 'p95': 0.10078}
- height: {'p5': 0.01806, 'p25': 0.02639, 'p50': 0.03472, 'p75': 0.04722, 'p95': 0.07361}
