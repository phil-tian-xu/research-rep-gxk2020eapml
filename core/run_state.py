"""Atomic model-year checkpoints and compatibility checks for annual runs."""

from contextlib import contextmanager
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from uuid import uuid4

import yaml

from utils.config_utils.config import CfgNode
from utils.path_info import root_path


def atomic_write(path, content):
    """Publish one complete UTF-8 file; abandoned temporary files are ignored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_identity():
    """Hash execution sources, including uncommitted code, without requiring Git."""
    root = Path(root_path)
    paths = [root / name for name in (
        "main.py", "core/run.py", "core/run_state.py", "core/splits.py",
        "core/model_selection.py", "core/model_inputs.py", "core/losses.py",
        "core/optimizers.py", "core/data_query/datasets.py", "utils/general_utils.py",
        "core/batch_run.py", "core/data_query/parquet_dataset.py",
        "core/dense_parquet_run.py", "core/dense_linear_run.py", "core/dense_glm_run.py", "core/data_examine/pcr_memory.py",
    )]
    for directory in ("core/models", "core/evaluation", "utils/config_utils"):
        paths.extend((root / directory).glob("*.py"))
    return {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "PyYAML", "iopath", "pyarrow")},
        "source_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(paths)},
    }


def resolve_run_dir(path):
    path = Path(path)
    return path if path.is_absolute() else Path(root_path) / path


def create_run(cfg, dataset_info):
    if not cfg.RUN.OUTPUT_DIR:
        raise ValueError("A resumable run requires RUN.OUTPUT_DIR.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = resolve_run_dir(cfg.RUN.OUTPUT_DIR) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "config": yaml.safe_load(cfg.dump()),
                "dataset": dataset_identity(dataset_info), "runtime": runtime_identity()}
    atomic_write(folder / "resolved_config.yaml", cfg.dump())
    atomic_write(folder / "dataset.json", json.dumps(dataset_info, indent=2, allow_nan=False))
    atomic_write(folder / "manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
    return folder


def dataset_identity(info):
    if not info or not all(info.get(key) for key in ("name", "panel_sha256", "metadata_sha256")):
        raise ValueError("Checkpointing requires a verified saved dataset and its hashes.")
    return {key: info[key] for key in ("name", "panel_sha256", "metadata_sha256")}


def load_run_config(folder):
    folder = resolve_run_dir(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported run checkpoint schema.")
    saved = yaml.safe_load((folder / "resolved_config.yaml").read_text(encoding="utf-8"))
    if saved != manifest["config"]:
        raise ValueError("Saved configuration differs from the run manifest.")
    return CfgNode(saved)


def validate_run(folder, cfg, dataset_info):
    manifest = json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "config": yaml.safe_load(cfg.dump()),
                "dataset": dataset_identity(dataset_info), "runtime": runtime_identity()}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Cannot resume: {key} differs from the saved run. Start a new run.")
    load_run_config(folder)  # Also detect edits to the readable config snapshot.


@contextmanager
def run_lock(folder):
    """OS lock released on exit or process death; the lock file may remain."""
    with (Path(folder) / ".run.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Run directory is in use: {folder}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def save_checkpoint(folder, metrics, predictions, tuning):
    path = Path(folder) / "years" / str(metrics["test_year"]) / f"{metrics['specification']}.json"
    if path.exists():
        raise FileExistsError(f"Completed checkpoint already exists: {path}")
    # R2 is undefined for an all-zero target; JSON null preserves that distinction.
    metrics = {key: (None if isinstance(value, float) and value != value else value)
               for key, value in metrics.items()}
    payload = json.dumps({"metrics": metrics, "predictions": predictions.to_dict(orient="list"),
                          "tuning": tuning}, allow_nan=False)
    envelope = {"sha256": hashlib.sha256(payload.encode()).hexdigest(), "payload": payload}
    atomic_write(path, json.dumps(envelope))


def read_checkpoints(folder, cfg, *, lazy=False):
    if lazy:
        return CheckpointIndex(folder, cfg)
    records = {}
    for path in sorted((Path(folder) / "years").glob("*/*.json")):
        record = read_checkpoint(path, cfg)
        metrics = record["metrics"]
        records[metrics["test_year"], metrics["specification"]] = record
    return records


def read_checkpoint(path, cfg):
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        if hashlib.sha256(payload.encode()).hexdigest() != envelope["sha256"]:
            raise ValueError("checksum mismatch")
        record = json.loads(payload)
        metrics, tuning = record["metrics"], record["tuning"]
        year, name = metrics["test_year"], metrics["specification"]
        if (name not in cfg.RUN.MODELS.ENABLED or
                not cfg.RUN.SPLIT.FIRST_TEST_YEAR <= year <= cfg.RUN.SPLIT.LAST_TEST_YEAR or
                path.parent.name != str(year) or path.stem != name or
                tuning["test_year"] != year or tuning["specification"] != name):
            raise ValueError("model/year identity mismatch")
        if metrics["r2_zero"] is None:
            metrics["r2_zero"] = float("nan")
        return record
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid completed checkpoint {path}: {error}") from error


class CheckpointIndex(Mapping):
    """Index paths only; release each decoded model-year after its caller uses it."""

    def __init__(self, folder, cfg):
        self.cfg = cfg
        self.paths = {}
        for path in sorted((Path(folder) / "years").glob("*/*.json")):
            try:
                year = int(path.parent.name)
            except ValueError as error:
                raise ValueError(f"Invalid checkpoint path: {path}") from error
            if (str(year) != path.parent.name or path.stem not in cfg.RUN.MODELS.ENABLED or
                    not cfg.RUN.SPLIT.FIRST_TEST_YEAR <= year <= cfg.RUN.SPLIT.LAST_TEST_YEAR):
                raise ValueError(f"Invalid checkpoint path: {path}")
            self.paths[year, path.stem] = path

    def __getitem__(self, key):
        return read_checkpoint(self.paths[key], self.cfg)

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)

    def __contains__(self, key):
        return key in self.paths
