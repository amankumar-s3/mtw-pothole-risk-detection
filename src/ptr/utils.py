"""Small shared helpers (seeding, file listing, CSV writing)."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch is optional for most scripts
        pass


def ensure_dir(p: str | Path) -> Path:
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


def list_files(root: str | Path, extensions: Iterable[str]) -> list[Path]:
    """Recursive, sorted, case-insensitive extension match."""
    exts = {e.lower() for e in extensions}
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in exts)


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write dict rows with a fixed header (header is written even when there are no rows)."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
