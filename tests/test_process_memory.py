from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.data_examine.csv_memory import process_memory, report_process_memory
from utils.memory_report import MemoryReport


class ProcessMemoryTests(unittest.TestCase):
    def test_linux_units_and_peak_survive_release(self):
        usage = SimpleNamespace(ru_maxrss=79752)
        resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: usage)
        with patch("sys.platform", "linux"), patch.dict(sys.modules, {"resource": resource}), \
                patch.object(Path, "read_text", side_effect=[
                    "Name:\tpython\nVmRSS:\t79752 kB\nVmHWM:\t79752 kB\n",
                    "VmRSS:\t14420 kB\nVmHWM:\t79752 kB\n"]):
            allocated, released = process_memory(), process_memory()
        self.assertEqual(allocated["resident_bytes"], 79752 * 1024)
        self.assertEqual(released["resident_bytes"], 14420 * 1024)
        self.assertEqual(released["peak_resident_bytes"], 79752 * 1024)
        self.assertIsNone(released["private_committed_bytes"])

    def test_linux_keeps_highest_os_counter(self):
        resource = SimpleNamespace(RUSAGE_SELF=0,
                                   getrusage=lambda who: SimpleNamespace(ru_maxrss=90))
        with patch("sys.platform", "linux"), patch.dict(sys.modules, {"resource": resource}), \
                patch.object(Path, "read_text", return_value="VmRSS: 50 kB\nVmHWM: 100 kB\n"):
            self.assertEqual(process_memory()["peak_resident_bytes"], 100 * 1024)

    def test_linux_reports_and_json_accept_missing_private_commit(self):
        counters = {"resident_bytes": 14420 * 1024, "peak_resident_bytes": 79752 * 1024,
                    "private_committed_bytes": None}
        output = StringIO()
        with TemporaryDirectory() as directory, redirect_stdout(output):
            with patch("core.data_examine.csv_memory.process_memory", return_value=counters):
                report_process_memory("Linux")
            monitor = MemoryReport(enabled=True)
            monitor.folder = Path(directory)
            with patch("utils.memory_report.process_memory", return_value=counters):
                monitor.snapshot("Final measurement")
            saved = json.loads(next(Path(directory).glob("memory_*.json")).read_text())
        self.assertEqual(saved["peak_resident_gib"], 79752 / 1024**2)
        self.assertIsNone(saved["stages"][0]["bytes"]["private_committed_bytes"])
        self.assertIn("unavailable on this OS", output.getvalue())
        self.assertIn("MAXIMUM OBSERVED PYTHON RAM:", output.getvalue())

    def test_missing_linux_proc_does_not_interrupt_optional_monitor(self):
        output = StringIO()
        with patch("sys.platform", "linux"), patch.object(Path, "read_text", side_effect=OSError("no proc")), \
                patch.dict(sys.modules, {"resource": SimpleNamespace()}), redirect_stdout(output):
            MemoryReport(enabled=True).snapshot("test")
        self.assertIn("Could not record measurement", output.getvalue())

    @unittest.skipUnless(sys.platform == "win32" or sys.platform.startswith("linux"),
                         "Live memory counters support Windows and Linux")
    def test_native_process_counters(self):
        counters = process_memory()
        self.assertGreater(counters["resident_bytes"], 0)
        self.assertGreaterEqual(counters["peak_resident_bytes"], counters["resident_bytes"])
        if sys.platform == "win32":
            self.assertGreater(counters["private_committed_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
