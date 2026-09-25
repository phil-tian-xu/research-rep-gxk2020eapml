"""Read a converted panel by return-year window, without assembling the panel."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class PanelBatch:
    X: np.ndarray | None
    y: np.ndarray
    row_position: np.ndarray
    return_month: np.ndarray


def load_window(panel, first_year, last_year):
    """Fill ordinary arrays from Parquet pieces; return the complete window."""
    window = panel.window(first_year, last_year)
    X = np.empty((window.n_rows, len(panel.features)), dtype=np.float64)
    y = np.empty(window.n_rows, dtype=np.float64)
    positions = np.empty(window.n_rows, dtype=np.int64)
    months = np.empty(window.n_rows, dtype="U7")
    start = 0
    for batch in window.batches():
        end = start + len(batch.y)
        X[start:end], y[start:end] = batch.X, batch.y
        positions[start:end], months[start:end] = batch.row_position, batch.return_month
        start = end
    if start != window.n_rows:
        raise ValueError("Loaded row count differs from the window metadata.")
    return X, y, positions, months


class ParquetPanel:
    def __init__(self, folder, *, batch_size=10000):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.folder = Path(folder).resolve()
        self.batch_size = batch_size
        raw = (self.folder / "metadata.json").read_bytes()
        self.metadata = m = json.loads(raw)
        if (m.get("schema_version") != 1 or m.get("format") != "parquet" or
                m.get("status") != "complete" or m.get("return_units") != "decimal"):
            raise ValueError("A complete float64 Parquet conversion manifest is required.")
        self.features = m["feature_columns"]
        roles = self.features + m["id_columns"] + [m[k] for k in
                ("target_column", "return_month_column", "feature_month_column", "row_position_column")]
        if not self.features or len(roles) != len(set(roles)):
            raise ValueError("Panel column roles must be unique and features nonempty.")
        counts, paths = Counter(), set()
        for part in m["parts"]:
            path = (self.folder / part["path"]).resolve()
            if not path.is_relative_to(self.folder) or path in paths or part["n_rows"] <= 0:
                raise ValueError("Invalid or duplicate Parquet part path/row count.")
            paths.add(path)
            counts[str(part["return_year"])] += part["n_rows"]
        if dict(counts) != m["year_counts"] or sum(counts.values()) != m["n_rows"]:
            raise ValueError("Parquet manifest row counts disagree.")
        self._verified = {}
        self.info = {"name": m["dataset_name"], "directory": str(self.folder),
                     "panel_sha256": m["source_panel_sha256"],
                     "metadata_sha256": hashlib.sha256(raw).hexdigest(), "metadata": m}

    def window(self, first_year, last_year):
        return YearWindow(self, first_year, last_year)

    def verify_part(self, part):
        path = self.folder / part["path"]
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if path in self._verified:
            if self._verified[path] != identity:
                raise ValueError(f"Parquet part changed during execution: {path}")
        else:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != part["sha256"]:
                raise ValueError(f"Parquet part SHA-256 mismatch: {path}")
            self._verified[path] = identity
        return path


class YearWindow:
    """A repeatable iterator; every pass opens only the selected return years."""

    def __init__(self, panel, first_year, last_year):
        if first_year > last_year:
            raise ValueError("Year window is empty.")
        self.panel = panel
        self.first_year, self.last_year = first_year, last_year
        self.parts = [part for part in panel.metadata["parts"]
                      if first_year <= part["return_year"] <= last_year]
        self.n_rows = sum(part["n_rows"] for part in self.parts)
        if not self.n_rows:
            raise ValueError(f"No observations in return years {first_year}-{last_year}.")

    def batches(self, *, features=True):
        m = self.panel.metadata
        target, date, feature_date, position = (m[k] for k in
            ("target_column", "return_month_column", "feature_month_column", "row_position_column"))
        columns = [target, date, feature_date, position]
        if features:
            columns += self.panel.features
        # Converted parts follow CSV order. Positions within each year must be
        # strictly increasing, even if the source interleaves different years.
        previous_by_year = {}
        for part in self.parts:
            path = self.panel.verify_part(part)
            count = 0
            with pq.ParquetFile(path) as parquet:
                numeric = [target] + (self.panel.features if features else [])
                if any(parquet.schema_arrow.field(c).type != pa.float64() for c in numeric):
                    raise ValueError(f"Model columns must be float64: {path}")
                if parquet.schema_arrow.field(position).type != pa.int64():
                    raise ValueError(f"Row positions must be int64: {path}")
                # Disable read-ahead/threaded column decoding to bound live buffers.
                for batch in parquet.iter_batches(batch_size=self.panel.batch_size,
                                                  columns=columns, use_threads=False):
                    frame = batch.to_pandas(use_threads=False)
                    months = pd.PeriodIndex(frame[date], freq="M")
                    feature_months = pd.PeriodIndex(frame[feature_date], freq="M")
                    if (months.isna().any() or feature_months.isna().any() or
                            not (months.year == part["return_year"]).all() or
                            not (feature_months + 1 == months).all()):
                        raise ValueError(f"Invalid return/feature month or partition year: {path}")
                    positions = frame[position].to_numpy(copy=True)
                    previous = previous_by_year.get(part["return_year"], -1)
                    if (len(positions) == 0 or positions[0] <= previous or
                            np.any(np.diff(positions) <= 0) or positions[-1] >= m["n_rows"]):
                        raise ValueError(f"Invalid/duplicate source row positions: {path}")
                    previous_by_year[part["return_year"]] = int(positions[-1])
                    y = frame[target].to_numpy(dtype=np.float64, copy=True)
                    X = frame[self.panel.features].to_numpy(dtype=np.float64, copy=True) if features else None
                    if not np.isfinite(y).all() or (features and not np.isfinite(X).all()):
                        raise ValueError(f"Nonfinite model values: {path}")
                    labels = months.astype(str).to_numpy()
                    count += len(y)
                    del frame, batch
                    yield PanelBatch(X, y, positions, labels)
            if count != part["n_rows"]:
                raise ValueError(f"Parquet part row count mismatch: {path}")
