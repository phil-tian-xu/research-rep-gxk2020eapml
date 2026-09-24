# Configured model runs

Update: `main.py` now saves completed model-years incrementally when OUTPUT_DIR
is set. Use `--stop-after-year` and `--resume` for staged execution; see
[Resumable Runs](Resumable%20Runs.md). The aggregate-only `save_run()` description
below still applies to direct Python callers, not the current main.py save path.

Implemented 2026-09-21 using the existing `CfgNode`, `get_cfg()`, and
`utils.general_utils.load_config()` unchanged. No registry is used. The previously
discussed config-framework improvements remain deferred as requested.

`main.py` is the single entry point for both synthetic checks and real replication:
load config -> load `DATA.DATASET` -> `core.run.run()` -> report -> optionally save.
There is no synthetic/real branch in the model runner. The former `core/model_demo.py`
entry point has been removed; `core/experiment.py` is now `core/run.py`, with
`run()`, `RunResult`, and `save_run()`. Config settings formerly under `EXPERIMENT`
now live under `RUN`; existing custom YAMLs must use the new key.

## Prepare data once, use one run entry point

Generate the two experimental datasets once:

```powershell
python -B -m core.data_query.generate_experimental_data
```

This writes `panel.csv` and `metadata.json` under each of:

- `data/experimental/synthetic_linear_v1/`
- `data/experimental/synthetic_nonlinear_v1/`

Both contain 3,056 observations (eight synthetic assets per month, March 1957
through December 1988), eight predictors, feature/return months, and decimal
excess-return targets. These are toy software-validation datasets, not the
paper's Monte Carlo designs or empirical data. Existing directories are never
overwritten. Re-running generation after these files exist raises an error.

The linear signal is `0.002 + X @ coefficients`. The nonlinear panel adds
`0.02 * (x0**2 - 1) + 0.015 * x1 * x2`. Both use the same predictors and noise
draws; the default noise SD is 0.05 and seed is 2020. Coefficients and all generator
settings are recorded in metadata. Predictor and noise streams are independent,
so extending the end month does not change already generated rows.

Generation settings live in `core/data_query/generate_experimental_data.py` and
its command-line arguments, not the run config. For example, its
`--seed`, `--start-month`, `--end-month`, and `--stocks-per-month` options control
fresh generation; different versions should use separate dataset identities.

Run either saved dataset through the same run entry point:

```powershell
python -B main.py --config model_demo
python -B main.py --config model_demo_nonlinear
```

The config name refers to `config/model_demo.yaml`, without the extension.
The command loads a clone of `config/default.py`, merges that YAML, loads the
saved panel selected by `DATA.DATASET`, and runs the annual model run. It prints a row for every
specification/year and a pooled out-of-sample summary for every specification.

The copied `config/demo.yaml` remains an example from the earlier project. Its
settings are not used by the model runner. The original default sections are
preserved. `DATA.DATASET` selects the panel and `RUN` controls the models
and run periods. Generator settings belong only to dataset preparation and metadata.
The currently implemented estimators are deterministic. `GENERAL.TASK_TYPE: "run"`
is descriptive metadata; it does not select a separate execution path.

YAML files override only relevant defaults. `model_demo_nonlinear.yaml` inherits
`model_demo.yaml` and changes just the run name and dataset. Changing model
settings never regenerates or modifies the input panel.

## Configuration responsibilities

| Section | Controls |
|---|---|
| `DATA.DATASET` | Saved modeling dataset name |
| `RUN.SPLIT` | First training year, first/last test years, rolling validation length |
| `RUN.MODELS.ENABLED` | Every specification to fit, tune, and report |
| `RUN.MODELS.FIT_INTERCEPT` / `STANDARDIZE` | Shared estimator preprocessing choices |
| `RUN.MODELS.LINEAR_SOLVER` | Proximal solver iterations, tolerance, initial step |
| `RUN.MODELS.OLS_HUBER` | Explicit training Huber threshold |
| `RUN.MODELS.ENET_HUBER` | Training threshold, rho, penalty candidate grid |
| `RUN.MODELS.PCR` / `PLS` | Component candidate grids; PLS numerical tolerance |
| `RUN.EVALUATION.VALIDATION` | Loss used to choose hyperparameters within each specification |
| `RUN.EVALUATION.TEST` | Loss reported for final predictions |
| `RUN.OUTPUT_DIR` | Optional parent directory for saved runs |

The example enables OLS, OLS_HUBER, ENET_HUBER, PCR, and PLS for 1987-1988.
Each year starts fresh fits. Training expands and validation rolls according to
`annual_splits`; validation/test observations never enter that year's fitting.
Parameters are not refitted using combined training and validation samples.

The workflow retains `tuning_result` for the tuning bundle and `tuned_model`
for its selected estimator. Every enabled specification is evaluated separately.
There is no overall winner that replaces the other specifications.

Explicit Python mappings translate names to constructors. Estimators still accept
ordinary keyword arguments, rather than knowing about CfgNode or run settings.
The registry module remains available in the repository but is not imported.

## Losses and metrics

Shared mathematical code is in `core/losses.py` and `core/optimizers.py`.
Public prediction losses have `loss_fn(y_true, y_pred) -> float` interfaces.
Residual-based training kernels use the same formulas. No numerical objective
changed as part of this move.

Set `NAME: "mse"` with `DELTA: null`, or `NAME: "huber"` with a positive `DELTA`.
The runner translates these settings to a callable and its display label.
The evaluation line remains:

```python
test_loss = test_loss_fn(observed, prediction)
```

For example, a separate run YAML could inherit the supplied demo:

```yaml
_BASE_: "model_demo.yaml"
GENERAL:
  NAME: "model_demo_huber_evaluation"
RUN:
  EVALUATION:
    TEST:
      NAME: "huber"
      DELTA: 0.1  # Illustrative, not an author-recovered threshold.
```

Changing test loss changes reported errors, not fitted models or selected
hyperparameters. Changing validation loss can change the chosen hyperparameters.
Changing a specification's training Huber threshold changes its fitted objective.

Pooled performance is computed from all concatenated test observations for each
specification. In particular, pooled R-squared is `1 - total_SSE / sum(y_test**2)`;
it is not the arithmetic mean of annual R-squared. A zero denominator yields NaN.
Pooled losses weight observations equally, including when annual sample sizes differ.

## Loading saved panels and preparing real data

`core/data_query/datasets.py` holds an ordinary name-to-path mapping; no registry
is involved. Loading returns features, targets, target months, and source metadata:

```python
from core.run import run
from core.data_query.datasets import load_dataset
from utils.general_utils import load_config

cfg = load_config("your_config")
X, y, return_month, dataset_info = load_dataset(cfg.DATA.DATASET)
run_result = run(X, y, return_month, cfg, dataset_info=dataset_info)
```

`X` is `(n, p)`, `y` is `(n,)`, and `return_month` has one realized target month per
row. DataFrames retain their feature names. Date/target positions must match the
feature rows. The prediction output includes `row_position` for joining back to
the input panel's identifiers, plus specification, test year, date, observed return,
and predicted return. It does not invent security identifiers from row numbers.

Each dataset directory has a `panel.csv` and `metadata.json`. Metadata specifies
`schema_version: 1`, `dataset_name`, `return_units: "decimal"`, `id_columns`,
`feature_month_column`, `return_month_column`, `target_column`, ordered
`feature_columns`, `n_rows`, and `panel_sha256`. Generated metadata additionally
records the generator settings and creation time.

The loader verifies the panel hash and row count, column roles, finite modeling
inputs, unique asset/target-month keys, and one-month feature-to-target alignment.
Identifier/date/target columns cannot be included as predictors. It preserves
file row order and selects predictors in the metadata's declared order.
It never downloads, prepares, or generates a missing dataset automatically.

The `gkx` mapping reserves `data/processed/gkx/` for a future prepared panel using
this same schema. That dataset has not been built; selecting it currently raises
a missing-dataset error. It does not reinterpret raw CRSP or Green data as a ready
modeling panel. Loading large prepared panels is still in memory, and checksum
verification adds a sequential file read.

This runner assumes the panel already handles financial availability lags, eligible
stocks, ranks/imputation, macro interactions, and excess-return construction. It does
not load or alter the repository's raw datasets. Current execution is in memory and
has not been benchmarked at full GKX scale.

## Saved results

Leave `RUN.OUTPUT_DIR: ""` for an in-memory run. To save, use a path such as
`"results/model_demo"`. Relative paths resolve against the project root. Each run
gets a fresh UTC timestamp subdirectory, containing:

- `resolved_config.yaml`: merged defaults and run settings.
- `dataset.json`: selected dataset identity, content/metadata hashes, and source metadata.
- `annual_metrics.csv`: every specification/year's counts and performance.
- `summary.csv`: pooled out-of-sample results per specification.
- `predictions.csv`: all test predictions and their input row positions.
- `tuning_history.json`: selected parameters and validation scores of every candidate.
- `environment.json`: Python and relevant package versions.

`run()` itself returns an `RunResult` without writing files.
`main.py` initializes a checkpoint directory when the output path is configured
and saves after each model-year. It rebuilds these combined reports after each
invocation; direct Python callers may still use `save_run()` for aggregate export.
Fitted estimator checkpoints are not saved in this stage.
For direct-array calls to `run()` without `dataset_info`, `dataset.json`
is null; such a call has no verified saved-dataset provenance.

## Replication choices and dependencies

The example YAML's Huber threshold and small grids, and the separate data generator, are examples.
They are not author settings or a real-data replication. Defaults leave empirical
thresholds/grid values unresolved until explicitly supplied by a run config.
Intercept/scaling, threshold construction, empirical grids, and input data still
need the research choices documented in the existing model notes.

The unchanged loader requires PyYAML and iopath. They are installed and declared
in `pyproject.toml`; tabulate is unnecessary because the registry is unused.
No config-loader hardening or replacement was performed.

Verification uses the existing numerical tests plus configured annual runs,
candidate-grid propagation, test-target independence, configurable losses,
pooled metric calculations, and round trips through saved run artifacts.
