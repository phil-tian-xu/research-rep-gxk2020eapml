This file supplements `AGENTS.md` with details for this replication. Sections marked “To be added” are not yet specified.

## Original Paper
- **Title:** Empirical Asset Pricing via Machine Learning
- **Authors:** Shihao Gu, Bryan Kelly, and Dacheng Xiu
- **Paper:** `docs/main_paper.pdf`
- **Online Appendix:** `docs/online_appendix.pdf`
## Replication Scope
To be added. 

## Data
Yearly Parquet conversion: `python -B -m core.data_cleaning.partition_gkx` reads
the prepared GKX CSV in 10,000-row chunks and writes `data/processed/gkx_parquet`.
Splits chunks by return year; retains float64 model values, audit columns, and
original row positions. Verifies each part by read-back and publishes a complete
manifest only at the end. Existing destinations are refused; raw/CSV data are
unchanged. Phil has completed the full conversion (3,740,946 rows, 920 features).
The v4 workflow now uses these partitions through a batch reader. See
`docs/working_notes/Parquet Conversion.md` for commands, integrity, and limitations.

CSV memory inspection: `python -B -m core.data_examine.csv_memory --rows 10000`.
Omit `--rows` to attempt the full prepared GKX CSV; `--csv` selects another file.
Uses neighboring prepared-panel metadata for the workflow's read_csv columns and
string dtypes. Reports disk size, DataFrame deep memory by dtype/column, and
Windows process resident/peak RAM and private committed memory separately.
Verified on 1,000 real rows; the full CSV was not loaded in this verification.

Standalone panel preparation: `python -B -m core.data_cleaning.prepare_gkx`.
Reads the four replication inputs under `data/raw` and writes the existing loader's
`data/processed/gkx` format. See `docs/working_notes/GKX Panel Preparation.md`
for alignment, explicit preprocessing choices, limitations, and run options.
The preparation script is not part of the main model workflow.

SIC inspection: `python -B -m core.raw_data_examinine.data_check_sic` reads the
entire `data/raw/datashare.csv` (no replication-period filter) and writes
`data/sic_inspection/`: `sic_values.csv`, `permno_sic_summary.csv`,
`missing_sic_rows.csv`, and `summary.csv`. Missingness statuses refer only to
observations in that file. Dates retain the original GKX DATE labels; numeric
codes including 99 are reported as supplied, without imputation or recoding.

Missing-SIC decision agreed on 2026-09-21: retain affected stocks, preserve
`sic2` as NA in source/audit data, and encode missing industry with all 74 SIC
dummies zero. Retain every industry dummy; do not drop a reference category.
An audit flag `sic_missing` must be excluded from model features. Do not fill
SIC over time or add a missing-industry predictor in the baseline. This is our
replication choice, not verified GKX author treatment. See the "Missing SIC:
agreed baseline and follow-up" section of `docs/working_notes/GKX Panel Preparation.md`.
Implemented in `prepare_gkx.py`: missing SIC is retained, original `sic2` and
`sic_missing` are audit-only panel columns, and monthly reports count missing
SIC before and after target exclusions. Invalid nonmissing SIC still fails.
Verified with a March-April 1957 real-data build and the unchanged dataset loader;
the full 1957-2016 wide panel has not yet been built in this verification.

## Other Requirements
To be added.

## Development experiments
PCR SVD correction (2026-09-25): the SCRP run completed 1987–2010 but failed
in 2011 at SciPy's 32-bit matrix-index guard. PCR now uses NumPy economy SVD;
Phil confirmed SCRP NumPy has OpenBLAS USE64BITINT (ILP64). Full-sized 2011
verification and migration of the saved run remain pending. Resume hash checks
are unchanged. See `docs/working_notes/Full Universe PCR Workflow.md`.

Local PLS development: `python -B main.py --config gkx_pls_p05_v1 --memory`
uses the fixed 5% Parquet sample, original complete-array PLS, K=0..10, and
1987–2016 annual testing. Supports stop/resume and existing Table 1/Figure 3
reports. The shared dense runner supports one PCR or PLS specification per run;
the full-window loader is in `core/data_query/parquet_dataset.py`.
See `docs/working_notes/PLS Small Sample Workflow.md`. Keep the active SCRP PCR
checkout unchanged until its run/resumes finish because execution hashes are strict.

SCRP production launcher: `run_pcr_scrp.sh`, with deployment config
`gkx_pcr_scrp` inheriting v5 and changing only the output directory to
`/data/users/s1155269217/GKX2020/results/gkx_pcr_full_v5`. Requests one node,
8 CPUs, 192G RAM and two days; uses `.venv/bin/python`, capped BLAS threads
and GNU time for Linux peak RSS. Supports `--stop-after-year` and `--resume`.
Model fitting is unchanged. See `docs/Running Python on SCRP.md` for the
1987-1996, 1997-2006 and 2007-2016 submissions, logs and output files.
This launcher has not been submitted to SCRP by the agent.

Current preference: use available machine RAM (64 GB installed), rather than
designing around an 8-GiB target. Prefer complete annual training arrays and
the original estimator; do not introduce batch QR fitting for this baseline.
The earlier batch workflow remains separate; v4 still selects that older path.

Full-universe version of the simple experiment:
`python -B -m core.data_examine.pcr_full_memory`. Reuses the same dense experiment
function with `data/processed/gkx_parquet` and separate `results/pcr_full_memory`
outputs. Defaults to a single 1987 test window and fixed K=10; `--test-year`
selects another window. No RAM cap. Full-data execution is left to Phil;
Phil completed the default 1987 full-universe run at
`results/pcr_full_memory/20260923T153943483633Z`: fixed K=10,
475,906 training rows, peak resident 16.715481 GiB. Later windows remain unmeasured.

Main-run dense PCR configuration: `gkx_pcr_full_v5` uses the full Parquet panel,
1987-2016 and the agreed K=0..100 grid. RUN.FITTING=dense dispatches to
`core/dense_parquet_run.py`: original PCR SVD on complete float64 training arrays,
release training, validate candidates, release validation, test and checkpoint.
Existing annual resume and Table 1 All/Figure 3 reporting are retained.
`main.py --memory` optionally prints/saves process RAM; it is not a config
parameter or RAM cap, and may be toggled on resume. Legacy v4 still uses batch QR.
143 tests passed, including dense equivalence, staged execution, diagnostic
toggle and PCR-only reports. The 2016 dense training window has 5.9 times the
1987 rows; roughly extrapolated peak ~98 GiB is not yet measured and may exceed
64 GB. See `docs/working_notes/Full Universe PCR Workflow.md`.

Simple PCR-only experiment: `python -B -m core.data_examine.pcr_memory`.
Loads the complete 5% Parquet training window into float64 NumPy arrays, calls
the original PCRRegressor.fit once, releases training arrays, then loads complete
validation and test windows sequentially. Default test year 1987, training
1957-1974, validation 1975-1986; fixed K=10 is a demonstration setting (no tuning).
No added QR reduction, batch optimization, or RAM target. Reports exact X/y
array bytes and OS process peak separately. Real run completed at
`results/pcr_memory/20260923T152603763633Z`: 22,395 x 920 training features;
X+y=0.153674 GiB, peak resident RAM=0.993053 GiB. Validation MSE matches the
saved original dense K=10 candidate exactly. Test R2=0.456731%.
Three focused tests pass. See `docs/working_notes/Simple PCR Experiment.md`.

Pre-run verification tool: `python -B -m core.data_examine.verify_batch` has
`compare` (saved 5% checkpoints versus newly fitted Parquet results) and `memory`
(full-universe first/largest annual windows in fresh Python processes). Default
memory target is peak resident RAM strictly below 8 GiB, not an enforced cap.
The largest v4 window is test year 2016: training 1957-2003, 2,796,159 rows.
Reference is `results/gkx_table1_figure3_v1/20260922T084211035119Z` (30 years).
Requires a separate conversion of the exact 5% CSV; source hashes are checked.
Production fitting is unchanged by this diagnostic addition. The real 5% comparison
`compare_20260923T053443957803Z` completed 1987 and 1988, then stopped on a
1988 OLS_HUBER prediction discrepancy (max abs 0.0168642591, RMSE difference
0.00640390857 in decimal-return units). Other recorded comparisons passed;
cause remains unresolved. Peak worker resident RAM was 0.391117 GiB (1987)
and 0.387238 GiB (1988); full-universe peak memory remains unverified.
Verifier summaries now distinguish INCOMPLETE from EXCEEDED, expose the max
observed peak, and track completed versus selected windows. `summary --run-dir`
reads existing reports without fitting; workers print live OS peak RAM and
save memory_latest.json approximately every 30 seconds. Eleven focused
verification tests pass after this reporting fix. No fitting/tolerances changed.
See `docs/working_notes/Batch Verification.md` for commands, tolerances and scope.

Full-universe comparison: `gkx_table1_figure3_v4` now uses `gkx_parquet`,
DATA.FORMAT=parquet and BATCH_SIZE=10000, with PCR, PLS, and OLS_HUBER for
1987-2016. Estimation/evaluation settings match the v1 pilot.
Requested execution stages end in 1996, 2006, and 2016; use `--resume` for
later stages. Start a new run after this execution-code/backend change.
`core/batch_run.py` fits annual windows using training-only blocked QR and
small-matrix PCR/SIMPLS; Huber preserves the training residual quantile and
svd_apg objective with repeated batch passes. No full feature/score matrix is
retained. Checkpoints remain per model-year; reports load one model-year at a
time. See `docs/working_notes/Batch Parquet Workflow.md` for methods and limits.
Verification: 126 tests pass. Batch versus dense fits and staged versus uninterrupted results
agree in synthetic tests. Real 1957 training (10,671 rows, 920 features) with
10,000 real 1987 prediction rows passed all three models; peak process resident
RAM was 0.563 GiB, Huber converged in 8 iterations. This is a bounded smoke test,
not a full 1987 or 30-year replication. Original Huber empirical instability
remains unresolved; no regularization or clipping was introduced.

Milestone 1 is achieved: Phil confirmed staged and uninterrupted real-data runs
produce the same results. Milestone 2 uses `gkx_table1_figure3_v1`: PCR/PLS/OLS_HUBER,
5% sample, test years 1987-2016, PCR K=0..100, PLS K=0..10, intercept on and no
additional standardization. Phil approved K=0 as a training-mean forecast.
The paper specifies Huber's 99.9% quantile but not its construction. After checking
the paper and public author simulation code, Phil explicitly approved the 99.9th
percentile of absolute training OLS residuals, recomputed annually. This is an
interpretation, not a recovered author rule. OLS_HUBER uses the new opt-in
`svd_apg` solver; other configs retain their solver behavior. PCR tuning reuses
one training SVD. Saved-result diagnostics are produced by
`python -B -m core.replication_report --run-dir RUN_DIRECTORY` and automatically
after each successful stage of the new config. See
`docs/working_notes/Table 1 and Figure 3 Pilot.md` for sources, choices, and outputs.
Verification: 110 tests pass. The real 1987 pilot completed in run
`results/gkx_table1_figure3_v1/20260922T084211035119Z`; PCR selected 14 and PLS 2.
OLS_HUBER converged but has extremely poor test performance (-2378.18% R2)
and extreme forecasts. Its empirical instability remains unresolved; do not
treat solver convergence or this one-year run as milestone 2 achievement.

`config/gkx_all_models_workflow_v2.yaml` extends the PLS-only v1 workflow:
same 5% development dataset, training from 1957, rolling 12-year validation,
and five annual test years 1987-1991. Currently enables only OLS, PCR, and PLS;
OLS_HUBER and ENET_HUBER are temporarily disabled after OLS_HUBER failed to
converge even at 50,000 iterations. Implementations and inactive settings are retained.
Intercept on, standardization
off; PCR/PLS components [1,2,3], Huber delta 0.1, elastic-net rho 0.5 and
penalties [0.001,0.01] are development choices, not verified paper settings.
The linear solver budget is 50,000 iterations with tolerance 1e-8: the first
real-data attempt hit the 5,000-iteration limit for unpenalized Huber regression.
The runner rejects unconverged candidates rather than evaluating them.
Run with `python -B main.py --config gkx_all_models_workflow_v2`;
timestamped outputs go under `results/gkx_all_models_workflow_v2/`.
Candidate progress is flushed to the console. Fitting/validation exceptions retain
their type and traceback with model, test year, candidate, stage, and parameter notes;
nonconvergence includes iterations, stationarity, tolerance, and stopping threshold.
Test prediction/scoring errors include model/year and selected parameters.

Resumable execution is implemented in `core/run_state.py`, `core/run.py`, and
`main.py`. Runs with OUTPUT_DIR save each completed model-year atomically;
`--stop-after-year YEAR` limits an invocation and `--resume RUN_DIRECTORY`
loads the saved config and skips completed units. Full horizon stays in config.
Resume checks config, panel/metadata hashes, execution source hashes, and Python/
package versions. Changed estimation code/settings require a new run. Incomplete
fits restart; old aggregate-only runs cannot resume. See
`docs/working_notes/Resumable Runs.md` for the workflow_v2 2+2+1 commands.
Verification: 101 tests passed, including three-stage versus uninterrupted
workflow_v2 settings on a small synthetic panel, skipped fits, partial-year
failure recovery, interrupted writes, corruption, compatibility, and locking.
This verification did not execute the five-year real-data experiment.

## Development samples
`python -B -m core.data_cleaning.sample_gkx` generates `gkx_dev_p05_seed2020`
and `gkx_dev_p10_seed2020` from the processed full panel. Select PERMNOs once
using a fixed seed and retain every parent observation for selected securities;
the 5% universe is nested in 10%. Do not resample by month or require full-period
survival. Preserve feature values and all parent dates. Both dataset names are
registered in the existing loader. See `docs/working_notes/GKX Development Samples.md`.

## Evaluation implementation
Reusable methods under `core/evaluation` cover Table 1 pooled zero-benchmark R2
and monthly size subsamples; Figure 3 complexity counts; and training-sample
grouped zero-out importance plus averaging/normalization/ranking for Figures 4/5
and Table 4. Table 3's Diebold-Mariano interface is a deferred placeholder.
The annual runner is not yet wired to the new methods. Group mappings and
negative-score normalization are explicit caller choices; exact author grouping
and aggregation conventions remain unverified. See
`docs/working_notes/Evaluation Methods.md` for contracts and integration notes.
