"""Logging helpers: console + optional per-run file, run id in every line."""
from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

LOGGER_NAME = "ptr"
_FMT = "%(asctime)s | %(levelname)-7s | %(run_id)s | %(name)s | %(message)s"


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def make_run_id() -> str:
    """Timestamp + short random suffix, e.g. 20260101_120000_ab12cd."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def setup_logging(run_dir: Optional[Path] = None, run_id: str = "-", level: int = logging.INFO) -> logging.Logger:
    """Configure the ``ptr`` logger (idempotent: previous handlers are replaced)."""
    logger = logging.getLogger(LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    logger.setLevel(level)
    logger.propagate = False
    flt = _RunIdFilter(run_id)
    fmt = logging.Formatter(_FMT)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.addFilter(flt)
    logger.addHandler(sh)
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(flt)
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
