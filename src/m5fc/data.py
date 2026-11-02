"""Loading M5 data and building the 12-level aggregation hierarchy.

Series are held as a wide matrix ``Y`` of shape (n_series, n_days) plus an
``ids`` frame describing each row. Wide matrices make aggregation, seasonal
naive and metric computation plain numpy operations; ``to_long`` converts to
the long (unique_id, ds, y) format that statsforecast expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HORIZON = 28
SEASON = 7

# The M5 standard levels: (level number, name, grouping columns).
LEVELS: list[tuple[int, str, list[str]]] = [
    (1, "total", []),
    (2, "state", ["state_id"]),
    (3, "store", ["store_id"]),
    (4, "category", ["cat_id"]),
    (5, "department", ["dept_id"]),
    (6, "state_category", ["state_id", "cat_id"]),
    (7, "state_department", ["state_id", "dept_id"]),
    (8, "store_category", ["store_id", "cat_id"]),
    (9, "store_department", ["store_id", "dept_id"]),
    (10, "item", ["item_id"]),
    (11, "state_item", ["state_id", "item_id"]),
    (12, "item_store", ["item_id", "store_id"]),
]
LEVEL_NAMES = {lvl: name for lvl, name, _ in LEVELS}
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class Hierarchy:
    """Series values for one or more aggregation levels."""

    ids: pd.DataFrame  # columns: unique_id, level
    Y: np.ndarray  # (n_series, n_days) float32
    dates: pd.DatetimeIndex  # length n_days

    def level(self, lvl: int) -> "Hierarchy":
        mask = (self.ids["level"] == lvl).to_numpy()
        return Hierarchy(self.ids[mask].reset_index(drop=True), self.Y[mask], self.dates)

    def slice_days(self, end: int) -> "Hierarchy":
        """Keep days [0, end) — used to cut a training window at a forecast origin."""
        return Hierarchy(self.ids, self.Y[:, :end], self.dates[:end])


def load_sales(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, np.ndarray]:
    sales = pd.read_parquet(data_dir / "sales_train_evaluation.parquet")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    return sales[ID_COLS], sales[day_cols].to_numpy(dtype=np.float32)


def load_calendar(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    cal = pd.read_parquet(data_dir / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def build_hierarchy(levels: list[int] | None = None, data_dir: Path = DATA_DIR) -> Hierarchy:
    """Aggregate the 30,490 bottom series to the requested M5 levels (default: all 12)."""
    meta, bottom = load_sales(data_dir)
    cal = load_calendar(data_dir)
    dates = pd.DatetimeIndex(cal["date"].iloc[: bottom.shape[1]])
    levels = levels or [lvl for lvl, _, _ in LEVELS]

    id_parts, y_parts = [], []
    for lvl, _, cols in LEVELS:
        if lvl not in levels:
            continue
        if not cols:
            keys = pd.Series(["Total"])
            agg = bottom.sum(axis=0, keepdims=True)
        else:
            key = meta[cols].astype(str).agg("_".join, axis=1)
            codes, uniques = pd.factorize(key, sort=True)
            # Sparse 0/1 summing matrix: (n_groups x n_bottom) @ bottom.
            S = sparse.csr_matrix(
                (np.ones(len(codes)), (codes, np.arange(len(codes)))),
                shape=(len(uniques), len(codes)),
            )
            agg = S @ bottom.astype(np.float64)
            keys = pd.Series(uniques)
        id_parts.append(pd.DataFrame({"unique_id": f"L{lvl}/" + keys, "level": lvl}))
        y_parts.append(agg.astype(np.float32))

    ids = pd.concat(id_parts, ignore_index=True)
    return Hierarchy(ids, np.vstack(y_parts), dates)


def first_nonzero(Y: np.ndarray) -> np.ndarray:
    """Index of the first non-zero observation per row (n_days if the row is all zero).

    Leading zeros in M5 mostly mean "item not yet stocked", not "zero demand", so
    models are fit and metrics are scaled from the first sale onward (M5 convention).
    """
    nz = Y != 0
    return np.where(nz.any(axis=1), nz.argmax(axis=1), Y.shape[1])


def to_long(h: Hierarchy, trim_leading_zeros: bool = True, max_days: int | None = None) -> pd.DataFrame:
    """Convert to statsforecast's long format (unique_id, ds, y).

    ``max_days`` keeps only the most recent ``max_days`` observations of each series,
    which bounds ETS/ARIMA fitting cost on long histories.
    """
    n, T = h.Y.shape
    start = first_nonzero(h.Y) if trim_leading_zeros else np.zeros(n, dtype=int)
    if max_days is not None:
        start = np.maximum(start, T - max_days)
    # Keep at least two seasons so seasonal models can be fit.
    start = np.minimum(start, T - 2 * SEASON)
    lengths = T - start
    row = np.repeat(np.arange(n), lengths)
    col = np.concatenate([np.arange(s, T) for s in start])
    return pd.DataFrame(
        {
            "unique_id": pd.Categorical.from_codes(row, categories=h.ids["unique_id"]),
            "ds": h.dates[col],
            "y": h.Y[row, col],
        }
    )
