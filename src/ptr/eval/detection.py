"""Detection metrics computed by us (hand-checkable): P, R, F1 at a confidence, AP@0.5, AP@0.5:0.95,
F1-optimal confidence (choose it on VAL only), PR / F1 curves and a failure gallery.

AP uses all-point interpolation of the precision envelope with greedy per-image matching (a prediction matches the
best still-unmatched ground-truth box with IoU >= threshold). Numbers can differ slightly from Ultralytics' own
``val`` (101-point interpolation); never mix the two in one table.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image

from ..logging_setup import get_logger
from ..utils import list_files

log = get_logger("eval.detection")

Box = tuple[float, float, float, float]
Pred = tuple[float, Box]                       # (confidence, xyxy pixels)
PredictFn = Callable[[Path], list[Pred]]


def iou_xyxy(a: Box, b: Box) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_flags(preds_by_image: dict[str, list[Pred]], gts_by_image: dict[str, list[Box]],
                thr: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-prediction TP flags at one IoU threshold. Returns (confs desc, tp flags, n_gt)."""
    confs: list[float] = []
    flags: list[int] = []
    n_gt = sum(len(v) for v in gts_by_image.values())
    for img, preds in preds_by_image.items():
        gts = gts_by_image.get(img, [])
        used = [False] * len(gts)
        for conf, box in sorted(preds, key=lambda p: -p[0]):
            best, bj = thr, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = iou_xyxy(box, g)
                if v >= best:
                    best, bj = v, j
            confs.append(conf)
            if bj >= 0:
                used[bj] = True
                flags.append(1)
            else:
                flags.append(0)
    order = np.argsort(-np.asarray(confs), kind="stable") if confs else np.array([], dtype=int)
    return np.asarray(confs, dtype=float)[order], np.asarray(flags, dtype=int)[order], n_gt


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP from TP flags sorted by descending confidence."""
    if n_gt == 0:
        return math.nan
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def prf_at_conf(confs: np.ndarray, tp: np.ndarray, n_gt: int, conf_thr: float) -> dict:
    keep = confs >= conf_thr
    tpn = int(tp[keep].sum())
    fpn = int(keep.sum()) - tpn
    fnn = n_gt - tpn
    p = tpn / (tpn + fpn) if (tpn + fpn) else math.nan
    r = tpn / n_gt if n_gt else math.nan
    f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else (0.0 if n_gt else math.nan)
    return {"tp": tpn, "fp": fpn, "fn": fnn, "precision": p, "recall": r, "f1": f1}


def best_f1_conf(confs: np.ndarray, tp: np.ndarray, n_gt: int, step: float, floor: float) -> tuple[float, float]:
    """(confidence, F1) maximising F1 on a grid. Call this with VAL predictions only.

    If F1 is flat over a range of confidences, the MIDDLE of that range is returned (the first maximum would be an
    arbitrary, extreme choice)."""
    grid, f1s = [], []
    c = floor
    while c <= 1.0 + 1e-9:
        f = prf_at_conf(confs, tp, n_gt, c)["f1"]
        if f == f:
            grid.append(c)
            f1s.append(f)
        c += step
    if not grid:
        return floor, math.nan
    best = max(f1s)
    plateau = [g for g, f in zip(grid, f1s) if f >= best - 1e-9]
    return round(float(np.median(plateau)), 6), best


def evaluate_predictions(preds: dict[str, list[Pred]], gts: dict[str, list[Box]], iou_thrs: Sequence[float],
                         conf: float) -> dict:
    out: dict = {"n_images": len(gts), "n_gt_boxes": sum(len(v) for v in gts.values()), "conf": conf}
    aps = []
    for thr in iou_thrs:
        c, t, n = match_flags(preds, gts, thr)
        aps.append(average_precision(t, n))
        if abs(thr - 0.5) < 1e-9:
            out["ap50"] = aps[-1]
            out.update(prf_at_conf(c, t, n, conf))
    out["ap50_95"] = float(np.nanmean(aps)) if aps and not all(a != a for a in aps) else math.nan
    return out


# ---------------------------------------------------------------------------------------------- data + prediction
def load_yolo_split(split_dir: Path, gt_class: int, exts: Sequence[str]) -> tuple[list[Path], dict[str, list[Box]]]:
    """Images of ``split_dir/images`` and ground-truth boxes (pixels) of class ``gt_class`` from ``labels``."""
    from ..dataprep import read_yolo_txt, yolo_to_xyxy

    imgs = list_files(split_dir / "images", exts)
    gts: dict[str, list[Box]] = {}
    for ip in imgs:
        with Image.open(ip) as im:
            w, h = im.size
        lp = split_dir / "labels" / f"{ip.stem}.txt"
        rows = read_yolo_txt(lp) if lp.exists() else []
        gts[ip.name] = [tuple(yolo_to_xyxy(x, y, bw, bh, w, h)) for c, x, y, bw, bh in rows if c == gt_class]  # type: ignore[misc]
    return imgs, gts


def predict_folder(predict: PredictFn, images: Sequence[Path]) -> dict[str, list[Pred]]:
    return {ip.name: list(predict(ip)) for ip in images}


# ---------------------------------------------------------------------------------------------- plots + gallery
def plot_curves(preds: dict, gts: dict, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, t, n = match_flags(preds, gts, 0.5)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    if len(c) and n:
        ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
        ax[0].plot(ctp / n, ctp / np.maximum(ctp + cfp, 1e-12))
        grid = np.linspace(0.0, 1.0, 101)
        ax[1].plot(grid, [prf_at_conf(c, t, n, g)["f1"] for g in grid])
    ax[0].set(xlabel="recall", ylabel="precision", title=f"PR @IoU0.5 ({title})", xlim=(0, 1), ylim=(0, 1.02))
    ax[1].set(xlabel="confidence", ylabel="F1", title="F1 vs confidence", xlim=(0, 1), ylim=(0, 1.02))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def failure_gallery(images: Sequence[Path], preds: dict, gts: dict, conf: float, out_dir: Path, max_n: int,
                    iou_thr: float = 0.5) -> list[dict]:
    """Save the images with the most false positives / false negatives (green TP, red FP, blue missed GT)."""
    import cv2

    scored = []
    for ip in images:
        p = [x for x in preds.get(ip.name, []) if x[0] >= conf]
        g = gts.get(ip.name, [])
        used = [False] * len(g)
        tpf = []
        for cf, b in sorted(p, key=lambda x: -x[0]):
            bj, best = -1, iou_thr
            for j, gb in enumerate(g):
                if not used[j] and iou_xyxy(b, gb) >= best:
                    bj, best = j, iou_xyxy(b, gb)
            if bj >= 0:
                used[bj] = True
            tpf.append((cf, b, bj >= 0))
        fp = sum(1 for _, _, ok in tpf if not ok)
        fn = used.count(False)
        if fp + fn:
            scored.append((fp + fn, fp, fn, ip, tpf, [gb for gb, u in zip(g, used) if not u]))
    scored.sort(key=lambda x: -x[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, (_, fp, fn, ip, tpf, missed) in enumerate(scored[:max_n], start=1):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        for cf, b, ok in tpf:
            col = (0, 200, 0) if ok else (0, 0, 230)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
            cv2.putText(img, f"{cf:.2f}", (int(b[0]), max(12, int(b[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        for b in missed:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (230, 120, 0), 2)
        name = f"{rank:02d}_fp{fp}_fn{fn}_{ip.stem}.jpg"
        cv2.imwrite(str(out_dir / name), img)
        rows.append({"file": name, "source_image": ip.name, "false_positives": fp, "false_negatives": fn})
    return rows


# ---------------------------------------------------------------------------------------------- orchestration
def run_evaluation(name: str, splits: dict[str, Path], predict: PredictFn, gt_class: int, exts: Sequence[str],
                   dcfg: dict, out_dir: Path, conf_from: Optional[str], fixed_conf: Optional[float],
                   gallery_split: Optional[str] = None) -> dict:
    """Evaluate ``predict`` on every split; the operating confidence comes from ``conf_from`` (VAL) only."""
    data: dict[str, tuple] = {}
    for sname, sdir in splits.items():
        imgs, gts = load_yolo_split(sdir, gt_class, exts)
        preds = predict_folder(predict, imgs)
        data[sname] = (imgs, gts, preds)
        log.info("%s: %d images, %d boxes", sname, len(imgs), sum(len(v) for v in gts.values()))
    note = ""
    if fixed_conf is not None:
        conf = fixed_conf
        note = f"confidence {conf} fixed by the user (not tuned on any evaluated split here)"
    elif conf_from and conf_from in data:
        _, g, p = data[conf_from]
        c, t, n = match_flags(p, g, 0.5)
        conf, f1v = best_f1_conf(c, t, n, dcfg["f1_conf_grid_step"], dcfg["pred_conf_floor"])
        note = f"confidence {conf} = F1-optimal on '{conf_from}' only (F1 {f1v:.3f}); applied unchanged to the other splits"
    else:
        raise ValueError("no operating confidence: pass --conf, or include the split named by --conf-from")
    results = {}
    for sname, (imgs, gts, preds) in data.items():
        results[sname] = evaluate_predictions(preds, gts, dcfg["iou_thresholds"], conf)
        plot_curves(preds, gts, out_dir / f"detection_{name}_{sname}_curves.png", sname)
    gallery = []
    if gallery_split and gallery_split in data:
        imgs, gts, preds = data[gallery_split]
        gallery = failure_gallery(imgs, preds, gts, dcfg["gallery_conf"], out_dir / f"gallery_{name}_{gallery_split}",
                                  dcfg["gallery_max"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["split", "n_images", "n_gt_boxes", "conf", "tp", "fp", "fn", "precision", "recall", "f1", "ap50", "ap50_95"]
    with open(out_dir / f"detection_{name}.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for s, r in results.items():
            fh.write(",".join([s] + [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols[1:]]) + "\n")
    lines = [f"# Detection evaluation: {name}", "", f"- {note}", f"- ground-truth class id evaluated: {gt_class}",
             "- metrics computed by `ptr.eval.detection` (all-point AP, greedy matching); see its docstring", "",
             "| split | images | boxes | conf | precision | recall | F1 | AP@0.5 | AP@0.5:0.95 |", "|---|---|---|---|---|---|---|---|---|"]
    for s, r in results.items():
        lines.append(f"| {s} | {r['n_images']} | {r['n_gt_boxes']} | {r['conf']:.2f} | {r['precision']:.3f} | "
                     f"{r['recall']:.3f} | {r['f1']:.3f} | {r['ap50']:.3f} | {r['ap50_95']:.3f} |")
    if "test_a" in results and "test_b" in results:
        a, b = results["test_a"], results["test_b"]
        lines += ["", f"Test A vs Test B gap (AP@0.5): {a['ap50'] - b['ap50']:+.3f} "
                      f"(Test A {a['n_images']} images / {a['n_gt_boxes']} boxes; Test B {b['n_images']} images / "
                      f"{b['n_gt_boxes']} boxes). Report this gap as a finding; do not hide it."]
    if any(r["n_gt_boxes"] < 100 for r in results.values()):
        lines += ["", "WARNING: a split has fewer than 100 ground-truth boxes: treat its numbers as indicative only."]
    if gallery:
        lines += ["", f"Failure gallery ({gallery_split}): `gallery_{name}_{gallery_split}/` ({len(gallery)} images)."]
    (out_dir / f"detection_{name}.md").write_text("\n".join(lines) + "\n")
    return {"results": results, "conf": conf, "note": note, "gallery": gallery}
