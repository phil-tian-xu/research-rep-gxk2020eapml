"""Optional Windows/Linux process memory observations; never changes estimation."""

from datetime import datetime, timezone
import json

from core.data_examine.csv_memory import process_memory
from core.run_state import atomic_write


class MemoryReport:
    def __init__(self, *, enabled=False):
        self.enabled = enabled
        self.folder = None
        self.stages = []
        self.stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def snapshot(self, stage, *, array_bytes=None):
        if not self.enabled:
            return
        try:
            counters = process_memory()
            self.stages.append({"stage": stage, "bytes": counters, "array_bytes": array_bytes})
            if counters:
                peak = counters["peak_resident_bytes"] / 1024**3
                print(f"[memory] {stage}: Python RAM={counters['resident_bytes'] / 1024**3:.3f} GiB; "
                      f"MAX SO FAR={peak:.3f} GiB", flush=True)
                if array_bytes is not None:
                    print(f"[memory] Complete X+y arrays={array_bytes / 1024**3:.3f} GiB", flush=True)
                if stage == "Final measurement":
                    print(f"MAXIMUM OBSERVED PYTHON RAM: {peak:.3f} GiB", flush=True)
            else:
                print("[memory] OS process counters unavailable.", flush=True)
            if self.folder is not None:
                atomic_write(self.folder / f"memory_{self.stamp}.json", json.dumps({
                    "scope": "Current process only, excluding children; lifetime resident peak; each resume starts a new process.",
                    "peak_resident_gib": counters["peak_resident_bytes"] / 1024**3 if counters else None,
                    "stages": self.stages}, indent=2))
        except Exception as error:
            # Optional diagnostics must not prevent fitting or mask its exception.
            print(f"[memory] Could not record measurement: {error}", flush=True)
