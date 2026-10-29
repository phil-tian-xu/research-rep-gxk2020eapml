# Compact feature experiments

## Interface and compatibility

Use the existing launcher with `--dataset NAME` and optional `--config NAME`.
The dataset manifest declares predictor names and order. The model sees the same
array interface, and NN input width follows the supplied matrix automatically.
The existing `--sample full/p05` shortcuts still select the original datasets.
Do not pass both selectors. `list`, `status`, `resume`, `combine` and `collect`
continue to use saved experiment identities and resolved settings.

The launcher and Parquet loader validate the declared compact/expanded feature definitions before fitting. A stale expanded dataset with eight extra standalone macros, an incorrect compact feature order, or a missing feature specification is rejected. Original dataset names retain their existing metadata contract.

The [implementation audit](Compact%20Feature%20Audit.md) records comparisons against the pre-change commit, a complete 5% data audit, and real 176-feature runs.

| Dataset | Inputs with the full 74-industry vocabulary | Security sample |
| --- | --- | --- |
| `gkx_parquet` | Original 920: 94 characteristics, 752 macro products, 74 SIC dummies | Full |
| `gkx_dev_p05_seed2020_parquet` | Original 920 | Existing fixed 5% sample |
| `gkx_compact_full` / `gkx_compact_p05` | 176: 94 characteristics, 8 standalone macros, 74 SIC dummies | Full / existing fixed 5% |
| `gkx_expanded_full` / `gkx_expanded_p05` | Paper's original 920: 94 characteristics, 752 macro products, 74 SIC dummies | Full / existing fixed 5% |

The expanded dataset copies the paper's original inputs exactly, including their
column order and values. It adds no standalone macros and does not recompute the
existing interaction products. A source with fewer industry categories produces
correspondingly fewer columns; preparation never
silently changes its vocabulary or requires 74 categories.

## Prepare the datasets once

From the repository root, with the existing source Parquet panels and raw macro
inputs present:

```powershell
.venv\Scripts\python.exe -B -m core.data_cleaning.feature_gkx --dataset gkx_compact_p05
.venv\Scripts\python.exe -B -m core.data_cleaning.feature_gkx --dataset gkx_expanded_p05
```

Use `gkx_compact_full` and `gkx_expanded_full` for full-sample versions. The source
panel is selected from the name: the existing fixed-5% panel for `_p05`, and the
full panel for `_full`. `--raw-dir PATH` and `--batch-size N` are optional.

Preparation reads bounded batches and preserves identifiers, audit columns,
targets, dates and source row positions. Expanded preparation copies all original
predictors and needs no raw macro files. Compact preparation projects existing
ranked characteristics and industry indicators, then adds eight standalone macros.
No security resampling, reranking, imputation or additional characteristic lag is
applied. The compact dataset's macros use the existing construction helper and
join on `feature_month`, one calendar month before `return_month`. That helper
also reads the existing risk-free file; targets remain copied from the panel.
Standalone macro names begin with `macro_`, so `macro_bm` and `macro_ep` remain
distinct from firm-level `bm` and `ep`.

Each written batch is read back and compared; source checksums, row counts,
date alignment and finite values are checked. New manifests record source panel
provenance and, for compact inputs, the added macro source. Existing output
directories are refused. An interrupted
build has no completion manifest and cannot be launched as a complete dataset.

Predictor scaling is **none**, matching the existing supplied-input NN convention.
Any future scaling must be an explicit, training-only, saved transformation with
a new experiment specification. Macro publication timing has the same documented
limitations as the original replication; this conversion does not audit data vintages.

## Select the experiment with arguments

```powershell
.venv\Scripts\python.exe -B run_experiment.py nn3 --local --dataset gkx_compact_p05 --years 1987 --config gkx_nn3_v1
.venv\Scripts\python.exe -B run_experiment.py nn3 --local --dataset gkx_expanded_p05 --years 1987 --config gkx_nn3_v1
.venv\Scripts\python.exe -B run_experiment.py ols --local --dataset gkx_compact_p05 --years 1987
.venv\Scripts\python.exe -B run_experiment.py ols3-mse --local --dataset gkx_compact_p05 --years 1987
```

`nn1` through `nn5` retain the current architectures, optimizer, tuning procedure,
early stopping and ensembles. A versioned `--config` can change their tuning grids
or other research settings. Both feature specifications tune separately on validation
data; using the same config supplies the same candidate grid, not fixed selected values.

`ols` fits ordinary unpenalized squared-error OLS with all declared dataset inputs.
`ols3-mse` fits only `mvel1`, `bm`, `mom12m` (or the three names explicitly supplied
in `RUN.MODELS.OLS3.FEATURES`). The existing `ols3` command keeps its Huber meaning.
Both new benchmarks reuse `LinearRegressor`, fit on the same training window only,
record validation loss without train+validation refitting, and save portable fit
state through the existing annual checkpoints. Their reports have no GKX Huber
reference value or tuned-complexity curve.

Resource flags and backends have the same meaning as in the primary run guide.
No feature preparation happens implicitly in a fitting command. `--dry-run`
prepares the saved plan without fitting or submitting a job.

Changing execution sources changes existing strict resume hashes. Use fresh
experiment IDs on this branch; retain the original checkout/environment for
unfinished historical runs. Source-hash guards have not been weakened.

## Read the comparison

Keep observations, chronology and evaluation identical across models. Compare
pooled/annual zero-benchmark out-of-sample R2, monthly forecast losses, training
diagnostics and computation. OLS versus NN assesses the chosen nonlinear model's
predictive benefit; compact versus expanded NN compares the reduced inputs with
the paper's original input specification. It does not establish perfect recovery
of all interactions. Choose grids, architectures and a practically acceptable
performance difference using validation
or a prespecified rule, then assess held-out results with uncertainty that respects
monthly dependence.
