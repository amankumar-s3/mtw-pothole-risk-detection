"""Shared dataset-preparation logic (VOC/YOLO parsing, conversion, cleaning, hashing, splitting, audit).

The scripts in scripts/ are thin argparse wrappers around these functions so that tests can call them directly.
Nothing here downloads data. Third-party data is never committed.
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from .logging_setup import get_logger
from .utils import list_files

log = get_logger("dataprep")

SKIP_TXT_NAMES = {"classes.txt", "labels.txt", "readme.txt", "license.txt", "obj.names"}


# ---------------------------------------------------------------------------------------------- parsing
@dataclass
class VocAnn:
    width: int
    height: int
    objects: list[tuple[str, float, float, float, float]]  # name, xmin, ymin, xmax, ymax (pixels)


def parse_voc(path: Path) -> VocAnn:
    root = ET.parse(path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for o in root.iter("object"):
        name = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        if bb is None:
            continue
        try:
            xs = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        objs.append((name, xs[0], xs[1], xs[2], xs[3]))
    return VocAnn(w, h, objs)


def read_yolo_txt(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():   # handles \r\n
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            out.append((int(float(parts[0])), *[float(x) for x in parts[1:5]]))
        except ValueError:
            continue
    return out


def write_yolo_txt(path: Path, rows: Iterable[tuple[int, float, float, float, float]]) -> None:
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows), encoding="utf-8")


def image_info(path: Path) -> tuple[int, int, Optional[int]]:
    """(width, height, exif_orientation). Raises on corrupt images."""
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        orient = None
        try:
            orient = im.getexif().get(0x0112)
        except Exception:
            orient = None
        return im.size[0], im.size[1], orient


def norm_name(s: str, case_insensitive: bool) -> str:
    s = " ".join(s.split())
    return s.lower() if case_insensitive else s


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> tuple[float, float, float, float]:
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def yolo_to_xyxy(xc: float, yc: float, bw: float, bh: float, w: float, h: float) -> tuple[float, float, float, float]:
    return ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> Optional[tuple[float, float, float, float]]:
    """Clamp to the image; None if the clamped box is empty."""
    x1, x2 = min(max(x1, 0.0), w), min(max(x2, 0.0), w)
    y1, y2 = min(max(y1, 0.0), h), min(max(y2, 0.0), h)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return x1, y1, x2, y2


def pair_images_and_annotations(root: Path, fmt: str, exts: Sequence[str]) -> tuple[dict[str, Path], dict[str, Path], int]:
    """Index images and annotation files by file stem. Returns (images, annots, n_duplicate_image_stems)."""
    imgs: dict[str, Path] = {}
    dups = 0
    for p in list_files(root, exts):
        if p.stem in imgs:
            dups += 1
            continue
        imgs[p.stem] = p
    ann_ext = {".xml"} if fmt == "voc" else {".txt"}
    anns: dict[str, Path] = {}
    for p in list_files(root, ann_ext):
        if fmt == "yolo" and p.name.lower() in SKIP_TXT_NAMES:
            continue
        anns.setdefault(p.stem, p)
    return imgs, anns, dups


# ---------------------------------------------------------------------------------------------- audit
def load_annotations(root: Path, fmt: str, exts: Sequence[str]) -> list[tuple[Path, list[tuple[str, float, float, float, float]], tuple[int, int]]]:
    """[(image, [(label, x1, y1, x2, y2) pixels], (w, h))] for every image that has an annotation file."""
    imgs, anns, _ = pair_images_and_annotations(root, fmt, exts)
    out = []
    for stem, ap in sorted(anns.items()):
        ip = imgs.get(stem)
        if ip is None:
            continue
        try:
            w, h, _ = image_info(ip)
        except (UnidentifiedImageError, OSError, SyntaxError):
            continue
        if fmt == "voc":
            v = parse_voc(ap)
            W, H = (v.width or w), (v.height or h)
            sx, sy = w / W, h / H
            boxes = [(n, a * sx, b * sy, c * sx, d * sy) for n, a, b, c, d in v.objects]
        else:
            boxes = [(str(c), *yolo_to_xyxy(x, y, bw, bh, w, h)) for c, x, y, bw, bh in read_yolo_txt(ap)]
        out.append((ip, boxes, (w, h)))
    return out


def audit_dataset(root: Path, fmt: str, exts: Sequence[str]) -> dict:
    """Counts, class histogram, relative box-size percentiles. Pure measurement, no assumptions."""
    imgs, anns, dups = pair_images_and_annotations(root, fmt, exts)
    corrupt, hist = [], Counter()
    rel_w, rel_h, n_boxes = [], [], 0
    imgs_with_ann = 0
    for stem, ip in imgs.items():
        try:
            w, h, _ = image_info(ip)
        except (UnidentifiedImageError, OSError, SyntaxError):
            corrupt.append(str(ip))
            continue
        ap = anns.get(stem)
        if ap is None:
            continue
        imgs_with_ann += 1
        if fmt == "voc":
            v = parse_voc(ap)
            W, H = (v.width or w), (v.height or h)
            for name, x1, y1, x2, y2 in v.objects:
                hist[name] += 1
                n_boxes += 1
                rel_w.append((x2 - x1) / W)
                rel_h.append((y2 - y1) / H)
        else:
            for c, _, _, bw, bh in read_yolo_txt(ap):
                hist[str(c)] += 1
                n_boxes += 1
                rel_w.append(bw)
                rel_h.append(bh)

    def pct(a: list[float]) -> dict[str, float]:
        if not a:
            return {}
        q = np.percentile(np.array(a), [5, 25, 50, 75, 95])
        return dict(zip(["p5", "p25", "p50", "p75", "p95"], [round(float(x), 5) for x in q]))

    return {
        "root": str(root), "format": fmt, "n_images": len(imgs), "n_annotation_files": len(anns),
        "n_boxes": n_boxes, "images_with_annotation": imgs_with_ann,
        "images_without_annotation": len([s for s in imgs if s not in anns]),
        "annotations_without_image": len([s for s in anns if s not in imgs]),
        "duplicate_image_stems": dups, "corrupt_images": corrupt, "class_histogram": dict(hist.most_common()),
        "rel_width_percentiles": pct(rel_w), "rel_height_percentiles": pct(rel_h),
    }


def audit_markdown(name: str, a: dict) -> str:
    L = [f"# Audit: {name}", "", "Measured by `scripts/verify_datasets.py` (no values were assumed). "
         "Licence, source URL and the visual audit must still be filled in by hand in docs/data_sources.md.", "",
         f"- root: `{a['root']}`", f"- format: {a['format']}", f"- images: {a['n_images']}",
         f"- annotation files: {a['n_annotation_files']}", f"- boxes: {a['n_boxes']}",
         f"- images without annotation: {a['images_without_annotation']}",
         f"- annotations without image: {a['annotations_without_image']}",
         f"- duplicate image file stems (ignored): {a['duplicate_image_stems']}",
         f"- corrupt images: {len(a['corrupt_images'])}", "", "## Class histogram", "", "| class | boxes |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in a["class_histogram"].items()] or ["| (none) | 0 |"]
    L += ["", "## Relative box size (fraction of image)", "", f"- width: {a['rel_width_percentiles']}",
          f"- height: {a['rel_height_percentiles']}", ""]
    if a["corrupt_images"]:
        L += ["## Corrupt images", ""] + [f"- {p}" for p in a["corrupt_images"][:50]] + [""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- convert
def convert_source(input_root: Path, out_dir: Path, source: str, fmt: str, class_map: dict, exts: Sequence[str],
                   keep_negative_ratio: float, exif_policy: str, seed: int, case_insensitive: bool = True,
                   max_rel_width: float | None = None, max_rel_area: float | None = None) -> dict:
    """Convert one source to the unified YOLO layout out_dir/{images,labels}; pothole = class 0, rest dropped."""
    if fmt not in ("voc", "yolo"):
        raise ValueError("fmt must be voc or yolo")
    rng = random.Random(seed)
    if out_dir.exists():
        shutil.rmtree(out_dir)                       # idempotent rebuild of this source only
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    cmap = {norm_name(str(k), case_insensitive): int(v) for k, v in class_map.items()} if fmt == "voc" else \
        {int(k): int(v) for k, v in class_map.items()}
    imgs, anns, dups = pair_images_and_annotations(input_root, fmt, exts)
    st: dict = Counter()
    dropped_classes: Counter = Counter()
    st["duplicate_image_stems"] = dups
    st["annotations_without_image"] = len([s for s in anns if s not in imgs])
    exif_flagged: list[str] = []
    for stem in sorted(imgs):
        ip = imgs[stem]
        ap = anns.get(stem)
        if ap is None:
            st["unlabeled_images_skipped"] += 1      # e.g. RDD2022 test folders have no labels
            continue
        try:
            w, h, orient = image_info(ip)
        except (UnidentifiedImageError, OSError, SyntaxError):
            st["corrupt_images_skipped"] += 1
            continue
        if orient not in (None, 1):
            exif_flagged.append(ip.name)
            if exif_policy == "skip":
                st["exif_rotated_skipped"] += 1
                continue
        rows: list[tuple[int, float, float, float, float]] = []
        if fmt == "voc":
            v = parse_voc(ap)
            W, H = (v.width or w), (v.height or h)            # coordinates refer to the XML size
            if (W, H) != (w, h):
                st["xml_size_differs_from_image"] += 1
            for name, x1, y1, x2, y2 in v.objects:
                cls = cmap.get(norm_name(name, case_insensitive))
                if cls is None:
                    dropped_classes[name] += 1
                    st["boxes_dropped_class"] += 1
                    continue
                b = clamp_box(x1, y1, x2, y2, W, H)
                if b is None:
                    st["boxes_dropped_invalid"] += 1
                    continue
                if b != (x1, y1, x2, y2):
                    st["boxes_clamped"] += 1
                rows.append((cls, *xyxy_to_yolo(*b, W, H)))
        else:
            for c, xc, yc, bw, bh in read_yolo_txt(ap):
                cls = cmap.get(c)
                if cls is None:
                    dropped_classes[str(c)] += 1
                    st["boxes_dropped_class"] += 1
                    continue
                b = clamp_box(*yolo_to_xyxy(xc, yc, bw, bh, 1.0, 1.0), 1.0, 1.0)
                if b is None:
                    st["boxes_dropped_invalid"] += 1
                    continue
                rows.append((cls, *xyxy_to_yolo(*b, 1.0, 1.0)))
        if max_rel_width is not None or max_rel_area is not None:
            kept = []
            for r in rows:
                too_wide = max_rel_width is not None and r[3] > max_rel_width
                too_big = max_rel_area is not None and r[3] * r[4] > max_rel_area
                if too_wide or too_big:
                    st["boxes_dropped_too_large"] += 1
                else:
                    kept.append(r)
            rows = kept
        if not rows:
            if rng.random() >= keep_negative_ratio:
                st["empty_images_dropped"] += 1
                continue
            st["negative_images_kept"] += 1
        name = f"{source}__{stem}"
        shutil.copy2(ip, out_dir / "images" / f"{name}{ip.suffix.lower()}")
        write_yolo_txt(out_dir / "labels" / f"{name}.txt", rows)
        st["images_written"] += 1
        st["boxes_written"] += len(rows)
    report = dict(st)
    report.update({"source": source, "format": fmt, "dropped_class_histogram": dict(dropped_classes),
                   "exif_orientation_flagged": len(exif_flagged), "exif_policy": exif_policy})
    (out_dir / "convert_report.json").write_text(json.dumps(report, indent=2))
    log.info("converted %s: %s", source, report)
    return report


# ---------------------------------------------------------------------------------------------- clean + hash
def clean_source(src_dir: Path, min_box_px: float, tiny_policy: str, exts: Sequence[str]) -> list[dict]:
    """Remove corrupt images, drop tiny boxes. Returns report rows. Idempotent."""
    report: list[dict] = []
    img_dir, lab_dir = src_dir / "images", src_dir / "labels"
    for ip in list_files(img_dir, exts):
        lp = lab_dir / f"{ip.stem}.txt"
        try:
            w, h, _ = image_info(ip)
        except (UnidentifiedImageError, OSError, SyntaxError):
            ip.unlink(missing_ok=True)
            lp.unlink(missing_ok=True)
            report.append({"file": ip.name, "source": src_dir.name, "action": "removed_corrupt_image", "detail": ""})
            continue
        if not lp.exists():
            continue
        rows = read_yolo_txt(lp)
        keep = [r for r in rows if min(r[3] * w, r[4] * h) >= min_box_px]
        n_drop = len(rows) - len(keep)
        if n_drop == 0:
            continue
        if not keep and tiny_policy == "drop_image_if_empty":
            ip.unlink()
            lp.unlink()
            report.append({"file": ip.name, "source": src_dir.name, "action": "removed_image_all_boxes_tiny",
                           "detail": f"{n_drop} boxes < {min_box_px}px"})
        else:
            write_yolo_txt(lp, keep)
            report.append({"file": ip.name, "source": src_dir.name, "action": "dropped_tiny_boxes",
                           "detail": f"{n_drop} boxes < {min_box_px}px"})
    return report


def phash_int(path: Path, hash_size: int) -> int:
    import imagehash

    with Image.open(path) as im:
        h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
    return int("".join("1" if b else "0" for b in h.hash.flatten()), 2)


def hamming_clusters(hashes: Sequence[int], threshold: int) -> list[int]:
    """Union-find clusters of items whose hashes differ by <= threshold bits. Returns a cluster id per item."""
    n = len(hashes)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n:
        # 64-bit hashes fit uint64; larger sizes are handled through python ints
        big = max(hashes).bit_length() > 64
        if not big:
            arr = np.array(hashes, dtype=np.uint64)
            lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
            for i in range(n - 1):
                x = (arr[i + 1:] ^ arr[i]).view(np.uint8).reshape(-1, 8)
                d = lut[x].sum(axis=1)
                for j in np.nonzero(d <= threshold)[0]:
                    ra, rb = find(i), find(i + 1 + int(j))
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    if bin(hashes[i] ^ hashes[j]).count("1") <= threshold:
                        ra, rb = find(i), find(j)
                        if ra != rb:
                            parent[max(ra, rb)] = min(ra, rb)
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots]


def compute_clusters(interim_root: Path, hash_size: int, threshold: int, exts: Sequence[str]) -> list[dict]:
    """Cluster ALL images of ALL sources jointly (so cross-source duplicates share a cluster)."""
    items: list[tuple[str, Path]] = []
    for sd in sorted(p for p in interim_root.iterdir() if (p / "images").is_dir()):
        items += [(sd.name, ip) for ip in list_files(sd / "images", exts)]
    hashes = [phash_int(ip, hash_size) for _, ip in items]
    cid = hamming_clusters(hashes, threshold)
    nhex = max(1, (hash_size * hash_size + 3) // 4)
    return [{"file": ip.name, "source": s, "cluster_id": c, "phash": f"{h:0{nhex}x}"}
            for (s, ip), h, c in zip(items, hashes, cid)]


# ---------------------------------------------------------------------------------------------- splits
def assign_splits(clusters: list[dict], test_a_source: str, val_fraction: float, seed: int) -> tuple[dict[str, str], dict]:
    """file-key (source/file) -> split. Test A = the whole held-out source; the rest is split by hash cluster."""
    rng = random.Random(seed)
    key = lambda r: f"{r['source']}/{r['file']}"
    test_clusters = {r["cluster_id"] for r in clusters if r["source"] == test_a_source}
    split: dict[str, str] = {}
    info = {"leak_removed": 0}
    pool: list[dict] = []
    for r in clusters:
        if r["source"] == test_a_source:
            split[key(r)] = "test_a"
        elif r["cluster_id"] in test_clusters:
            info["leak_removed"] += 1                 # near-duplicate of a Test A image: excluded everywhere
        else:
            pool.append(r)
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in pool:
        groups[r["cluster_id"]].append(r)
    totals = Counter(r["source"] for r in pool)
    target = {s: val_fraction * n for s, n in totals.items()}
    val_count: Counter = Counter()
    gids = sorted(groups)
    rng.shuffle(gids)
    for g in gids:
        members = groups[g]
        major = Counter(m["source"] for m in members).most_common(1)[0][0]
        to_val = val_count[major] < target[major]
        for m in members:
            split[key(m)] = "val" if to_val else "train"
        if to_val:
            val_count[major] += len(members)
    return split, info


def check_no_leak(manifest: list[dict], test_a_source: str) -> list[str]:
    """Problems found in a split manifest (empty list = OK). Used by make_splits and by the tests."""
    problems = []
    by_cluster: dict[str, set[str]] = defaultdict(set)
    for r in manifest:
        by_cluster[str(r["cluster_id"])].add(r["split"])
        if r["split"] != "test_a" and r["source"] == test_a_source:
            problems.append(f"test_a source image outside test_a: {r['file']}")
        if r["split"] == "test_a" and r["source"] != test_a_source:
            problems.append(f"non-test_a source in test_a: {r['file']}")
    for c, s in by_cluster.items():
        if len(s) > 1:
            problems.append(f"cluster {c} spans splits {sorted(s)}")
    return problems


def write_splits(interim_root: Path, processed_root: Path, clusters: list[dict], split: dict[str, str],
                 exts: Sequence[str]) -> list[dict]:
    for s in ("train", "val", "test_a"):
        d = processed_root / s
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir(parents=True)
    manifest = []
    for r in clusters:
        k = f"{r['source']}/{r['file']}"
        sp = split.get(k)
        if sp is None:
            continue
        src_img = interim_root / r["source"] / "images" / r["file"]
        src_lab = interim_root / r["source"] / "labels" / (Path(r["file"]).stem + ".txt")
        shutil.copy2(src_img, processed_root / sp / "images" / r["file"])
        if src_lab.exists():
            shutil.copy2(src_lab, processed_root / sp / "labels" / src_lab.name)
        manifest.append({"file": r["file"], "source": r["source"], "split": sp, "cluster_id": r["cluster_id"]})
    return manifest


def split_stats(processed_root: Path, manifest: list[dict], exts: Sequence[str]) -> dict:
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"images": 0, "boxes": 0}))
    sizes: dict[str, list[float]] = defaultdict(list)
    for r in manifest:
        lp = processed_root / r["split"] / "labels" / (Path(r["file"]).stem + ".txt")
        rows = read_yolo_txt(lp) if lp.exists() else []
        c = counts[r["source"]][r["split"]]
        c["images"] += 1
        c["boxes"] += len(rows)
        sizes[r["split"]] += [math.sqrt(max(bw * bh, 0.0)) for _, _, _, bw, bh in rows]
    return {"counts": {s: dict(v) for s, v in counts.items()}, "sizes": dict(sizes)}


def write_dataset_yaml(path: Path, processed_root: Path) -> None:
    import yaml

    doc = {"path": str(processed_root), "train": "train/images", "val": "val/images", "test": "test_a/images",
           "names": {0: "pothole"}}
    path.write_text("# Written by scripts/make_splits.py. Colab: the notebook rewrites `path`.\n"
                    + yaml.safe_dump(doc, sort_keys=False))


# ---------------------------------------------------------------------------------------------- Test B tooling
def extract_frames(video: Path, clip_id: str, start_s: float, end_s: Optional[float], interval_s: float,
                   out_dir: Path, blur_threshold: float, jpeg_quality: int) -> dict:
    """Sample frames about every ``interval_s`` seconds, skipping blurry ones (Laplacian variance)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        raise ValueError("video reports no fps; cannot sample by time")
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = blurry = 0
    next_t = start_s
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t = idx / fps
        if t < next_t - 1e-9:
            continue
        if end_s is not None and t > end_s + 1e-9:
            break
        next_t += interval_s
        lap = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if lap < blur_threshold:
            blurry += 1
            continue
        cv2.imwrite(str(out_dir / f"{clip_id}_{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        kept += 1
    cap.release()
    rep = {"clip_id": clip_id, "kept": kept, "skipped_blurry": blurry, "fps": fps}
    log.info("extract_frames %s", rep)
    return rep


def write_prelabels(image_dir: Path, out_dir: Path, predict, exts: Sequence[str]) -> dict:
    """``predict(path) -> [(class_id, conf, (x1, y1, x2, y2) pixels)]``; writes editable YOLO .txt pre-labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for ip in list_files(image_dir, exts):
        w, h, _ = image_info(ip)
        rows = []
        for cls, _conf, b in predict(ip):
            bb = clamp_box(*b, w, h)
            if bb is not None:
                rows.append((int(cls), *xyxy_to_yolo(*bb, w, h)))
        write_yolo_txt(out_dir / f"{ip.stem}.txt", rows)
        n_img += 1
        n_box += len(rows)
    return {"images": n_img, "boxes": n_box}


def split_clip_id(stem: str) -> str:
    """'<clip_id>_<frame>' -> clip_id (the clip id itself may contain underscores)."""
    head, _, tail = stem.rpartition("_")
    return head if head and tail.isdigit() else stem


def validate_yolo_folder(root: Path, exts: Sequence[str], allowed_ids: Sequence[int], min_box_px: float) -> dict:
    """Validate an exported annotation folder: ids in the allowed set, boxes in [0, 1] and not tiny."""
    imgs, anns, _ = pair_images_and_annotations(root, "yolo", exts)
    problems: list[str] = []
    n_boxes = 0
    per_class: Counter = Counter()
    pairs: list[tuple[Path, Path]] = []
    for stem, ip in sorted(imgs.items()):
        ap = anns.get(stem)
        if ap is None:
            problems.append(f"{ip.name}: no label file")
            continue
        try:
            w, h, _ = image_info(ip)
        except (UnidentifiedImageError, OSError, SyntaxError):
            problems.append(f"{ip.name}: corrupt image")
            continue
        ok = True
        for c, xc, yc, bw, bh in read_yolo_txt(ap):
            n_boxes += 1
            per_class[c] += 1
            if c not in allowed_ids:
                problems.append(f"{ap.name}: class id {c} not in {list(allowed_ids)}")
                ok = False
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1
                    and -1e-6 <= xc - bw / 2 and xc + bw / 2 <= 1 + 1e-6 and -1e-6 <= yc - bh / 2 and yc + bh / 2 <= 1 + 1e-6):
                problems.append(f"{ap.name}: box outside [0, 1]")
                ok = False
            elif min(bw * w, bh * h) < min_box_px:
                problems.append(f"{ap.name}: box smaller than {min_box_px}px")
                ok = False
        if ok:
            pairs.append((ip, ap))
    return {"problems": problems, "n_images": len(imgs), "n_valid_images": len(pairs), "n_boxes": n_boxes,
            "per_class": dict(per_class), "pairs": pairs}


def read_provenance(path: Optional[Path]) -> dict[str, dict]:
    if path is None or not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh) if r.get("clip_id")}


def import_test_b(annotations: Path, out_root: Path, provenance: Optional[Path], exts: Sequence[str],
                  allowed_ids: Sequence[int], min_box_px: float, min_videos: int) -> dict:
    """Validate and place annotated frames in out_root/{images,labels}; all frames of one clip stay together."""
    v = validate_yolo_folder(annotations, exts, allowed_ids, min_box_px)
    if v["problems"]:
        return {"ok": False, "problems": v["problems"], "n_images": v["n_images"]}
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    prov = read_provenance(provenance)
    clips: dict[str, dict] = {}
    for ip, ap in v["pairs"]:
        cid = split_clip_id(ip.stem)
        shutil.copy2(ip, out_root / "images" / ip.name)
        shutil.copy2(ap, out_root / "labels" / f"{ip.stem}.txt")
        c = clips.setdefault(cid, {"clip_id": cid, "n_images": 0, "n_potholes": 0, "n_mtw": 0,
                                   "source_url": prov.get(cid, {}).get("url", "TO VERIFY: not in PROVENANCE.csv")})
        c["n_images"] += 1
        for cls, *_ in read_yolo_txt(ap):
            c["n_potholes" if cls == 0 else "n_mtw"] += 1
    # manifest over EVERYTHING now in out_root (previous imports included)
    all_clips: dict[str, dict] = {}
    for lp in sorted((out_root / "labels").glob("*.txt")):
        cid = split_clip_id(lp.stem)
        c = all_clips.setdefault(cid, {"clip_id": cid, "n_images": 0, "n_potholes": 0, "n_mtw": 0,
                                       "source_url": prov.get(cid, {}).get("url", "TO VERIFY: not in PROVENANCE.csv")})
        c["n_images"] += 1
        for cls, *_ in read_yolo_txt(lp):
            c["n_potholes" if cls == 0 else "n_mtw"] += 1
    with open(out_root / "clips.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "source_url", "n_images", "n_potholes", "n_mtw"])
        w.writeheader()
        w.writerows(all_clips.values())
    urls = {c["source_url"] for c in all_clips.values() if not c["source_url"].startswith("TO VERIFY")}
    n_videos = len(urls) if urls else len(all_clips)
    return {"ok": True, "problems": [], "n_images": v["n_valid_images"], "n_boxes": v["n_boxes"],
            "per_class": v["per_class"], "n_clips": len(all_clips), "n_distinct_videos": n_videos,
            "videos_from_provenance": bool(urls), "min_videos": min_videos, "enough_videos": n_videos >= min_videos}


def leak_check(test_dir: Path, train_dirs: Sequence[Path], hash_size: int, threshold: int, exts: Sequence[str]) -> list[tuple[str, str]]:
    """Near-duplicate (perceptual hash) pairs between ``test_dir`` images and ``train_dirs`` images."""
    tests = [(p, phash_int(p, hash_size)) for p in list_files(test_dir, exts)]
    refs = [(p, phash_int(p, hash_size)) for d in train_dirs for p in list_files(d, exts)]
    hits = []
    for tp, th in tests:
        for rp, rh in refs:
            if bin(th ^ rh).count("1") <= threshold:
                hits.append((tp.name, rp.name))
    return hits


# ---------------------------------------------------------------------------------------------- agreement
def box_iou_xywhn(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def annotator_agreement(dir_a: Path, dir_b: Path, iou_thr: float) -> dict:
    """Greedy one-to-one matching of boxes of the same class (IoU >= iou_thr), per class. YOLO txt folders."""
    fa = {p.stem: p for p in dir_a.glob("*.txt") if p.name.lower() not in SKIP_TXT_NAMES}
    fb = {p.stem: p for p in dir_b.glob("*.txt") if p.name.lower() not in SKIP_TXT_NAMES}
    common = sorted(set(fa) & set(fb))
    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"a": 0, "b": 0, "matched": 0})
    for stem in common:
        ra, rb = read_yolo_txt(fa[stem]), read_yolo_txt(fb[stem])
        for c in {r[0] for r in ra} | {r[0] for r in rb}:
            A = [r[1:] for r in ra if r[0] == c]
            B = [r[1:] for r in rb if r[0] == c]
            stats[c]["a"] += len(A)
            stats[c]["b"] += len(B)
            pairs = sorted(((box_iou_xywhn(x, y), i, j) for i, x in enumerate(A) for j, y in enumerate(B)), reverse=True)
            ua, ub = set(), set()
            for v, i, j in pairs:
                if v < iou_thr:
                    break
                if i in ua or j in ub:
                    continue
                ua.add(i)
                ub.add(j)
                stats[c]["matched"] += 1
    per_class = {}
    for c, s in sorted(stats.items()):
        tot = s["a"] + s["b"]
        per_class[c] = {**s, "agreement_f1": (2 * s["matched"] / tot) if tot else math.nan}
    return {"n_common_files": len(common), "only_in_a": sorted(set(fa) - set(fb)), "only_in_b": sorted(set(fb) - set(fa)),
            "iou_threshold": iou_thr, "per_class": per_class}
