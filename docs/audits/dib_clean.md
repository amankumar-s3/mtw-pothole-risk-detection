# Audit: dib_clean

Measured by `scripts/verify_datasets.py` (no values were assumed). Licence, source URL and the visual audit must still be filled in by hand in docs/data_sources.md.

- root: `data/raw/dib_clean`
- format: voc
- images: 713
- annotation files: 713
- boxes: 1157
- images without annotation: 0
- annotations without image: 0
- duplicate image file stems (ignored): 0
- corrupt images: 0

## Class histogram

| class | boxes |
|---|---|
| pothole | 1156 |
| o | 1 |

## Relative box size (fraction of image)

- width: {'p5': 0.09884, 'p25': 0.19417, 'p50': 0.32058, 'p75': 0.53659, 'p95': 0.8342}
- height: {'p5': 0.0534, 'p25': 0.12379, 'p50': 0.23697, 'p75': 0.37717, 'p95': 0.6699}
