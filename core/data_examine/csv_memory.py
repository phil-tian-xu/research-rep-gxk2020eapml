"""Measure CSV loading and DataFrame memory, without fitting or converting to NumPy.

From the repository root:
    python -B -m core.data_examine.csv_memory --rows 10000
    python -B -m core.data_examine.csv_memory

The default is the full prepared GKX CSV. A neighboring metadata.json, when
present, supplies the same columns/string dtypes used by the workflow loader.
Process counters use the Windows API; DataFrame reporting works on other OSes.
"""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
from time import perf_counter

import pandas as pd

from utils.path_info import data_path


def process_memory():
    """Return Windows process counters in bytes; None when unavailable.

    peak_resident_bytes is the OS-maintained lifetime peak, so brief spikes
    between Python sampling points are included.
    """
    if os.name != "nt":
        return None

    # https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex
    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"resident_bytes": counters.WorkingSetSize,
            "peak_resident_bytes": counters.PeakWorkingSetSize,
            "private_committed_bytes": counters.PrivateUsage}


def report_process_memory(label):
    """Report resident RAM separately from committed virtual memory on Windows."""
    counters = process_memory()
    if counters is None:
        print(f"{label}: process counters are available on Windows only.", flush=True)
        return
    print(
        f"{label}: resident RAM={counters['resident_bytes'] / 1024**3:.3f} GiB; "
        f"process-lifetime peak RAM={counters['peak_resident_bytes'] / 1024**3:.3f} GiB; "
        f"private committed memory={counters['private_committed_bytes'] / 1024**3:.3f} GiB",
        flush=True,
    )


def main(csv_path, rows=None):
    csv_path = Path(csv_path).resolve()
    if rows is not None and rows <= 0:
        raise ValueError("--rows must be positive; omit it to load the entire CSV.")
    print(f"CSV: {csv_path}", flush=True)
    print(f"Entire CSV on disk: {csv_path.stat().st_size / 1024**3:.3f} GiB", flush=True)
    print(f"Row limit: {rows if rows is not None else 'none (full CSV)'}", flush=True)
    print(f"Python PID: {os.getpid()}", flush=True)

    options = {"float_precision": "round_trip", "nrows": rows}
    metadata_path = csv_path.with_name("metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = metadata["id_columns"] + [metadata["feature_month_column"], metadata["return_month_column"]]
        options["usecols"] = labels + [metadata["target_column"]] + metadata["feature_columns"]
        options["dtype"] = {column: "string" for column in labels}
        print("Using workflow column selection and dtypes from metadata.json.", flush=True)
    else:
        print("No metadata.json: reading all columns with pandas dtype inference.", flush=True)

    report_process_memory("Before read_csv")
    start = perf_counter()
    print("Loading CSV into one DataFrame...", flush=True)
    try:
        panel = pd.read_csv(csv_path, **options)
    except MemoryError:
        report_process_memory("After failed read_csv")
        print("Loading failed: no complete DataFrame is available to measure. Try --rows 10000.", flush=True)
        raise
    elapsed = perf_counter() - start
    report_process_memory("After read_csv")
    print(f"Loaded {len(panel):,} rows x {len(panel.columns):,} columns in {elapsed:.1f} seconds.", flush=True)

    # Measure existing columns; do not allocate a second matrix with to_numpy().
    column_bytes = panel.memory_usage(index=False, deep=True)
    index_bytes = panel.index.memory_usage(deep=True)
    total_bytes = int(column_bytes.sum() + index_bytes)
    details = pd.DataFrame({"dtype": panel.dtypes.astype(str), "bytes": column_bytes})
    details["MiB"] = details["bytes"] / 1024**2
    print(f"\nDataFrame deep memory estimate, including index: {total_bytes / 1024**3:.3f} GiB ({total_bytes:,} bytes)")
    print(f"Index alone: {index_bytes / 1024**2:.3f} MiB")
    print("\nColumn memory by dtype (excluding index):")
    print((details.groupby("dtype")["bytes"].sum() / 1024**3).rename("GiB").to_string())
    print("\nLargest 15 columns:")
    print(details.sort_values("bytes", ascending=False).head(15).to_string())
    print("\nDataFrame memory is an accounting estimate, not total process RAM.")
    print("Process RAM includes Python, libraries, buffers, and allocator-retained memory.")
    print("Private committed memory is not the same as resident physical RAM.")
    report_process_memory("After memory reporting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=data_path / "processed" / "gkx" / "panel.csv")
    parser.add_argument("--rows", type=int, help="Load only the first N rows; default loads the full CSV.")
    args = parser.parse_args()
    main(args.csv, args.rows)
