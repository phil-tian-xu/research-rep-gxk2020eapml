# Compact feature workflow audit

Audit date: 2026-10-29. Baseline: Git commit `de801f7968a1b5122ea3ee0993a919efba8bb1ea`, before the compact-feature changes. Scope: feature preparation, dataset selection/loading, original model/config preservation, annual execution, fitting, validation selection, checkpoints, prediction restoration, completed resume and collected reports.

## Original workflow comparison

The baseline source was extracted into an isolated directory under `results/compact_audit_20261029/baseline`. Original datasets were read in place without copying or modifying them. Both source versions used the same Python environment, CUDA device, data, seeds and two CPU threads. The audit went through the launcher and annual worker; the historical worker was dispatched in-process so its dataset registry could point to the existing read-only panel.

| Check | Scope | Result |
| --- | --- | --- |
| Legacy NN1 and NN3 | 1987, original fixed 5% panel, 920 predictors, both original learning rates, L1 values 0.00001 and 0.001, two seeds, eight-epoch cap | Predictions, validation results, selected parameters, all saved NN weights and BN buffers match exactly |
| Legacy `ols3` | 1987, original fixed 5% panel, original Huber settings | Predictions, coefficients, threshold, validation results and metrics match exactly |
| Explicit `gkx_expanded_p05` versus legacy `--sample p05` | NN3 with the same controlled settings | Predictions and fit/tuning state match exactly |
| Legacy `ols3` on compact versus original data | Original Huber settings and original three predictors | Predictions and fit/tuning state match exactly |
| Completed resume | Every audit case | Completed annual checkpoints were skipped and fitting was not repeated |

The original main entry point, in-memory runner, NN annual loop, estimators, losses, optimizer, splits, model selection and original model YAML files match the baseline after normalizing Windows line endings. The new OLS dispatch and feature-manifest checks are additions. Configuration comparison excludes output paths and the new, unused OLS3 settings node. Runtime metadata, timestamps and source hashes are expected to differ; numerical fit comparisons exclude elapsed time and GPU memory accounting.

The saved October 24 production NN1 run was also replayed with current code on its original 3,892 test observations. CPU restoration of its saved CUDA fit differs from the original forecasts by at most `3.53157520296965e-7`, consistent with the existing cross-device tolerance. Its saved results were read without modification.

## Compact data audit

All 182,367 compact 5% observations across 718 months were checked. Characteristics, industry indicators, targets, return months and row positions match the original source. Each of the eight standalone macros matches an independently constructed reference from the previous calendar month's raw observations exactly, with zero maximum absolute error for every macro. Products using size, book-to-market and momentum also match the original interaction columns exactly across all 718 months. This follows the repository's documented macro observation-month convention.

The compact panel has 176 predictors and the corrected expanded panel has 920. Full-universe compact data has not yet been prepared locally. The full builder uses the same checked transformation and the existing full Parquet source.

## Execution checks

All 259 unit and integration tests passed in 90.269 seconds. Added regression coverage rejects a stale expanded feature list, incorrect compact ordering, missing feature specifications and malformed compact manifests before reading model arrays. These checks apply in the launcher and the Parquet loader; original dataset names keep their prior contract.

Real 176-input NN1–NN5 runs completed for 1987 using two seeds, the two original learning rates, two L1 values and an eight-epoch cap. Ordinary OLS and squared-error OLS-3 also completed. Each case verified saved prediction reconstruction, data/feature order, reports and completed resume. NN checks used CUDA training and CPU restoration within `5e-6` absolute and `1e-6` relative tolerance.

An additional compact NN3 run used the normal six-candidate grid, ten seeds, maximum 100 epochs, patience five and batch size 10,000. It used 22,395 training, 37,680 validation and 3,892 test observations. Its 1987 test zero-benchmark R2 was approximately **-2.75572%**. This confirms execution of the production settings; the planned full-universe, full-period predictive comparison remains outstanding.

## Running the intended sample

The prepared local dataset `gkx_compact_p05` is the fixed 5% security sample. `--sample full` continues to select the original 920-feature full panel. For a full-universe 176-feature run, prepare and explicitly select `gkx_compact_full`:

```powershell
.venv\Scripts\python.exe -B -m core.data_cleaning.feature_gkx --dataset gkx_compact_full
.venv\Scripts\python.exe -B run_experiment.py nn3 --local --dataset gkx_compact_full --years 1987:2016 --config gkx_nn3_v1
```

Use fresh experiment IDs on this branch. Existing strict source/environment resume checks remain enabled; unfinished historical runs require their original checkout and environment.

## Local evidence

- [Source and run comparison](../../results/compact_audit_20261029/audit.json)
- [Complete compact data audit](../../results/compact_audit_20261029/data_audit.json)
- [Explicit dataset interface equivalence](../../results/compact_audit_20261029/interface_equivalence.json)
- [Historical NN1 forecast replay](../../results/compact_audit_20261029/historical_replay.json)
- [Full test log](../../results/compact_audit_20261029/unittest.log)

Audit scripts, the historical source snapshot, resolved configurations and worker logs are retained under `results/compact_audit_20261029/`. They are local generated artifacts; production data and prior experiment files are preserved.
