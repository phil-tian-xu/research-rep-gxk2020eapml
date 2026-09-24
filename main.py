

"""Run configured models on a prepared dataset: python main.py --config model_demo.

Synthetic datasets and the real replication use this same entry point.
DATA.DATASET selects the panel; RUN controls models, periods, and evaluation.
"""

import argparse

from core.data_query.datasets import load_dataset
from core.run import run
from core.run_state import create_run, load_run_config, resolve_run_dir
from utils.general_utils import load_config


def main(config_name=None, *, resume=None, stop_after_year=None, memory=False):
    from utils.memory_report import MemoryReport
    monitor = MemoryReport(enabled=memory)
    monitor.snapshot("Before dataset loading")
    try:
        return _main(config_name, resume=resume, stop_after_year=stop_after_year, monitor=monitor)
    finally:
        monitor.snapshot("Final measurement")


def _main(config_name=None, *, resume=None, stop_after_year=None, monitor):
    if resume is not None and config_name is not None:
        raise ValueError("Use either config_name or resume; resume loads the saved configuration.")
    cfg = load_run_config(resume) if resume is not None else load_config(config_name or "model_demo")
    if stop_after_year is not None and not cfg.RUN.SPLIT.FIRST_TEST_YEAR <= stop_after_year <= cfg.RUN.SPLIT.LAST_TEST_YEAR:
        raise ValueError("--stop-after-year must be within the configured test years.")
    if stop_after_year is not None and resume is None and not cfg.RUN.OUTPUT_DIR:
        raise ValueError("Staged execution requires RUN.OUTPUT_DIR so progress can be resumed.")
    print(f"Loading dataset: {cfg.DATA.DATASET}", flush=True)
    data_format = cfg.DATA.get("FORMAT", "csv")
    if data_format == "parquet":
        from core.data_query.datasets import DATASET_PATHS
        from core.data_query.parquet_dataset import ParquetPanel
        fitting = cfg.RUN.get("FITTING", "batch")
        if fitting == "dense":
            from core.dense_parquet_run import run_parquet
        elif fitting == "batch":
            from core.batch_run import run_parquet
        else:
            raise ValueError("RUN.FITTING must be 'dense' or 'batch'.")
        panel = ParquetPanel(DATASET_PATHS[cfg.DATA.DATASET], batch_size=cfg.DATA.BATCH_SIZE)
        dataset_info = panel.info
    elif data_format == "csv":
        X, y, return_month, dataset_info = load_dataset(cfg.DATA.DATASET)
    else:
        raise ValueError("DATA.FORMAT must be 'csv' or 'parquet'.")
    run_dir = (resolve_run_dir(resume) if resume is not None else
               create_run(cfg, dataset_info) if cfg.RUN.OUTPUT_DIR else None)
    if run_dir is not None:
        monitor.folder = run_dir
        print(f"Run directory: {run_dir}\nResume with: python -B main.py --resume \"{run_dir}\"", flush=True)
    monitor.snapshot("Dataset ready")
    if data_format == "parquet":
        options = {"memory_snapshot": monitor.snapshot} if fitting == "dense" else {}
        run_result = run_parquet(panel, cfg, checkpoint_dir=run_dir, stop_after_year=stop_after_year, **options)
    else:
        run_result = run(X, y, return_month, cfg, dataset_info=dataset_info,
                         checkpoint_dir=run_dir, stop_after_year=stop_after_year)
    print(f"Run: {cfg.GENERAL.NAME}; dataset={cfg.DATA.DATASET}")
    print(run_result.annual_metrics.to_string(index=False))
    print("\nPooled out-of-sample performance by specification:")
    print(run_result.summary.to_string(index=False))
    monitor.snapshot("Model run complete")
    if run_dir is not None and cfg.RUN.get("REPORT_TABLE1_FIGURE3", False):
        from core.replication_report import make_report
        make_report(run_dir)
    return run_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", help="Config name under config/, without .yaml; defaults to model_demo.")
    source.add_argument("--resume", help="Existing run directory; loads its saved experiment configuration.")
    parser.add_argument("--stop-after-year", type=int, help="Inclusive test year at which to stop this invocation.")
    parser.add_argument("--memory", action="store_true", help="Print and save optional process RAM measurements; no RAM cap.")
    args = parser.parse_args()
    main(args.config, resume=args.resume, stop_after_year=args.stop_after_year, memory=args.memory)
