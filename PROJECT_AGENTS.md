This file supplements `AGENTS.md` with details for this replication. Sections marked “To be added” are not yet specified.

## Progress tracker — Phil edits only
- `docs/Progress Tracker.md` is Phil's simple To do / In progress / Done tracker.
- Only Phil may create or modify it. Agents must not write, update, move, or delete it,
  including through scripts or automatic status synchronization.
- Agents may read it and suggest specific updates in chat when progress changes.
  If it is missing, provide a draft in chat for Phil to save; do not create it.
- Distinguish a completed computation from reviewed replication results. Do not
  assume that a submitted job has started or finished without evidence.

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

Combined bundle recovery (2026-10-29): `run_experiment.py collect COMBINED_ID
--repair-sources` restores missing/corrupt embedded files from intact originals
only when their bytes match the frozen SHA-256 inventory. Intact copied files
are retained; later original edits/new files are not synchronized. Original
sub-experiments, collection plans, IDs, provenance and scores remain unchanged.
Replacement is staged/verified, with rollback and preservation of previous copies
if rollback itself fails. Unexpected embedded files are refused rather than lost.
New/repaired bundles use schema 2 and short `sources/00`, `sources/01` paths;
schema 1 remains readable. Intact old bundles compact without original access.
`status`/`list` check inventories and show `sources=present`; `--verify` hashes
all embedded files and shows `sources=verified`, failing nonzero on corruption.
Incomplete bundles show `reports=incomplete` even when old summaries exist.
Missing inventory manifests require a complete backup. Shared datasets/environment
remain external. No fitting/orchestrator code or production results changed.
All 247 tests passed (77.835s), including frozen-byte repair, preserved originals,
schema-1 migration, unavailable/changed originals, rollback failures and relocation
with paths over 260 characters. Real local NN3-NN5 imports were checked read-only
and remain incomplete; no server repair/submission was performed. See the primary
run guide and SCRP/AI Hub command sheets for recovery and transfer verification.

Self-contained combined results (2026-10-27): collecting a combined ID now
copies complete source experiment folders into its own `sources/` directory,
with a SHA-256 file inventory published only after copy/source verification.
The original experiment contents, modification times and directory entries are
unchanged; existing worker locks are opened read-only and no source lock files
are created. Nothing is moved, deleted, archived or hidden. Existing ready
combined reports are upgraded by their next `collect` without changing scores,
frozen plans or original provenance. `year_sources.csv` uses paths relative to
the combined folder and retains `original_*` locations. A combined launch-record
storage marker detects a missing bundle instead of silently reverting to the
originals. Repeated collection verifies the stored snapshot; it does not sync
later source edits. Status, completed resume, collection and Figures 4/5 support
relocated/downloaded folders, including Linux paths read on Windows. Incomplete
runs still execute through their original sources; snapshots never submit fits.
Download the entire collected combined folder; the processed panel/environment
remain separate. See `docs/Running Experiments.md` and the cluster run guides.
No fitting/orchestrator sources, model settings, production results or remote
jobs changed. All 240 tests passed (66.034s), including 15 combined tests for
preservation, real tiny NN pooling, legacy upgrades, portable Figures 4/5,
failed/corrupt-copy retries, whole-bundle loss, checksum guards, busy sources and
recursive-copy destination rejection.

Combined experiments (2026-10-25): `run_experiment.py combine SOURCE1 SOURCE2
--years FIRST:LAST` registers disjoint source plans before or after fitting.
The returned combined ID supports status/resume/collect and Figures 4/5. Resume
preserves each source's resources, skips completed/active sources and rejects
uncertain submission receipts; resource changes use the original source ID.
`collect SOURCE1 SOURCE2 --years FIRST:LAST` also registers and collects a group.
The new `core/combined_experiments.py` validates fixed coverage, config/runtime/
dataset/features, checkpoint hashes, predictions/counts/dates, recomputed MSE/R2
and selected validation results. Pooled R2 sums errors and return squares across
years. Original fits stay in place; the combined descriptor references them.
Outputs include annual metrics, NN retained-member training diagnostics, source
provenance and validation summary. Zero updates/weights and poor performance are
diagnostic flags, not reasons to choose another fit. See the primary run guide.
No fitting sources, annual orchestration or fitting runtime hashes changed.
Verification: all 231 tests passed (43.762s), including actual tiny NN split/
unsplit pooled equivalence, mocked separate 3090/A100 submission pools, retry
guards, corrupt/missing/duplicate/incompatible sources, and Figures 4/5.
A real p05 1987/1988 normal-launcher smoke completed locally on the RTX 5080:
`results/combined_nn_smoke_20261025T151747000531Z/verification.json`.
It deliberately used one grid candidate, two seeds and three epochs solely to
check infrastructure; its poor scores are not replication results. Production
NN settings remain unchanged. No cluster jobs submitted. The tracker is Phil-only.

Selectable NN cluster GPUs (2026-10-25): `run_experiment.py` now accepts
`--gpu-type a100/rtx3090/rtx3060/any`; `--gpus 1` remains the per-year allocation
and `--parallel N` controls concurrent annual jobs / maximum total GPUs.
SCRP NN jobs use the new generic `run_nn_scrp.sh`: default A100 requests retain
partition/constraint `a100`, while RTX/any requests default to partition `scrp`
with no A100 constraint. Existing A100 script is preserved for legacy use.
GPU type and resources persist on resume; an explicit type change selects its
default partition unless `--partition` is supplied. Old NN records without the
new field retain their A100 default. AI Hub keeps its untyped GPU default;
explicit types pass through as typed GRES labels. Model code/settings unchanged.
Full SCRP NN1 command: `run_experiment.py nn1 --scrp --sample full --years
1987:2016 --gpu-type rtx3090 --gpus 1 --parallel 4 --cpus 8 --mem 96G --time
12:00:00`. Requests up to four GPUs, 32 CPUs and 384G CPU RAM, subject to live
quotas/availability; the 12-hour limit is per annual task. All 27 launcher tests
passed, including typed requests, unchanged research settings, legacy resumes,
partition changes and CPU/AI Hub dispatch. A real local dry-run prepared all
30 full-data annual configs and the expected four-job RTX 3090 submission.
No cluster job submitted during this change; full-data RTX memory/runtime and
GPU-node compatibility remain unmeasured. See `docs/Running Experiments.md`.
Phil also reported the SCRP NN1 p05/1987 A100 pilot completed and collected:
test R2=-3.463426%, versus local -2.367893%. The difference and remote training
diagnostics still need review; successful execution is not predictive replication.

SCRP NN A100 submission (2026-10-24): new `run_nn_a100_scrp.sh` is selected
automatically for NN1-NN5 `--scrp` launches. Defaults to partition `a100`, one
typed `a100` GPU and the documented `a100` node constraint. Logs GPU/driver and
PyTorch/CUDA information and verifies one visible GPU plus a real CUDA matrix
forward/backward pass before the unchanged annual worker. First cluster test:
`run_experiment.py nn1 --scrp --sample p05 --years 1987 --parallel 1 --cpus 4
--mem 8G --time 00:30:00` from the SCRP project root after `uv sync --extra nn`.
NN full-run initial RAM/time defaults remain 64G/12h. CPU and AI Hub dispatch
remain separate. All 23 launcher tests passed and a real local dry-run preserved
the pilot/model settings and generated the expected A100 request; no cluster job
was submitted. See `docs/Running Experiments.md` for the command and log locations.

Neural networks (2026-10-24): `run_experiment.py nn1` through `nn5` now use
the existing local/SCRP/AI Hub annual workflow. Model/training code is in
`core/models/neural_network.py`; annual selection in `core/nn_run.py`. Review
`docs/working_notes/Neural Networks.md` for the exact mean-half-MSE plus dense-kernel
L1 objective, initialization, BN-after-ReLU, Adam, early stopping, ensemble tuning,
precision, seeds, and explicit author-unverified implementation choices. Configs
`gkx_nn1_v1` through `gkx_nn5_v1` default to the fixed 5% sample and CUDA; install
the optional pinned dependency with `uv sync --extra nn` (PyTorch 2.13.0, CUDA 13.0).
Single-GPU Slurm requests are supported; NN concurrency defaults to one. Existing
CPU commands do not request GPUs. New execution hashes require fresh experiments;
preserve older remote environments/checkouts for their existing resumes.
All 220 tests passed. Real local RTX 5080 full-grid, ten-seed pilots for 1987:
`nn1_p05_1987-1987_20261024T131639296733Z`, test R2=-2.367893%;
`nn3_p05_1987-1987_20261024T132334325467Z`, test R2=-0.605856%.
Both learned nonconstant predictions and changed all selected seeds' parameters,
but neither beat zero forecasts in that test year; do not call this successful
predictive replication. Each selected LR=.01, L1=.001 (both grid upper endpoints).
CPU-restored states verified all 3,892 saved CUDA test forecasts within 3.532e-7
for NN1 and 1.550e-7 for NN3; real completed-year resumes skipped fitting.
No NN2/NN4/NN5 real-panel fits or remote NN submissions were run. Cluster numerical
checks and full-period/full-sample results remain pending. Phil-only tracker untouched.

OLS-3 threshold sensitivity (2026-09-28): separate
`python -B -m core.data_examine.ols3_thresholds --sample full --years 1987 1988 2011 2012`
compares .999 quantiles of training absolute OLS-3 residuals, absolute returns,
and absolute demeaned returns, plus ordinary OLS. Reuses unchanged loss/estimator
with identical settings; no model selection or validation refit. Reads three
features, saves annual fits and validation/test predictions, pooled All-stock
Table 1 comparisons and diagnostic plots under
`results/huber_threshold_sensitivity/<UTC timestamp>/`. Separate
`run_ols3_thresholds_aihub.sh` requests 4 CPUs/8G/two hours; no remote submission.
Three focused tests passed and real 1987 p05 check completed: baseline R2
0.345153%, alternatives 0.345669%/0.345332%, OLS 0.331997%. No production fitting
source/config or old results changed. No resume integration; full-data/AI Hub
runs and Bash verification pending. See `docs/working_notes/OLS3 Threshold Sensitivity.md`.

Experiment archival (2026-09-28): `run_experiment.py remove ID_OR_ABSOLUTE_PATH`
moves one inactive experiment to the sibling `<results-root-name>_archive` folder,
preserving its actual folder name and all saved records. Supports `--root` and
`--aihub`; archived folders disappear from normal listing. Checks launcher records,
Slurm inactivity and execution/collection locks; refuses symlinks and destination
collisions. Restore the exact original path before resume/collect. No fitting
code or remote experiments changed. See `docs/Running Experiments.md` for usage.

Figures 4/5 saved-state update (2026-09-28): current dense PCR/PLS checkpoints
now save original-panel slopes/intercept, feature order and selected transforms/
directions; RF saves portable numeric tree arrays. ENet/OLS-3 and GLM already
save sufficient state. Figure 4 now restores GLM splines and RF trees as well
as linear models, verifying saved test predictions before training zero-out
importance. New `core.figure5_report --figure4 PATH [PATH ...]` consumes complete
annual importance reports: rank within year, average ranks over years, sum
across models. Requires matching years/predictor groups/data identities; preserves
signed scores and ties. This completes the previously deferred saving work for
current dense launcher models. Old PCR/PLS/RF checkpoints cannot be upgraded
without refitting; legacy CSV/batch writers are unchanged. New source hashes
require fresh fitting runs. All 203 tests passed. Fresh 1987-1988 fixed-5% PLS
run `pls_p05_1987-1988_20260928T120917066756Z` completed; restored all 7,772 test
predictions exactly and generated both reports without refitting. Both K=0 fits
give zero importance and tied mean rank 48 for all 95 groups, as expected.
No server jobs submitted. See `docs/working_notes/Figure 4 Workflow.md`.

PLS specification revision (2026-09-28): new `run_experiment.py pls` runs use
`gkx_pls_v2`: training-only predictor standardization and K=0..29. This supersedes
the launcher default noted in older entries below; `gkx_pls_p05_v1` stays intact
for historical reproduction. Existing SIMPLS, fitted intercept, annual splits,
MSE selection, and no-refit behavior are unchanged. Scaling and grid are motivated
by public author simulation code, not recovered empirical settings. All 197 tests
passed. A local 1987-1988 fixed-5% end-to-end check completed: selected K=0 in
both years, pooled test R2=0.130248%. No server jobs were submitted. These small
sample results do not establish improved full-sample replication.
PLS parameter saving for Figures 4/5 was deferred at that stage and has now
been implemented in the saved-state update above. See
`docs/working_notes/PLS Small Sample Workflow.md` for commands and evidence.

Figure 4 cluster launch scripts (2026-09-28): `run_figure4_scrp.sh` and
`run_figure4_aihub.sh` pass report arguments to `core.figure4_report`, using one
CPU-only Slurm job (initial 4 CPUs, 8G, four hours) for all selected years. SCRP
uses the recorded pg/scrp/c32g4 settings; AI Hub uses site defaults unless sbatch
options override them. Logs, environment and GNU time statistics go under
`results/figure4_logs/`; report output remains timestamped with a figure4 folder.
See `docs/working_notes/Figure 4 Workflow.md` for deployment and commands. No
remote jobs submitted; Bash/Slurm runtime verification remains pending because
Bash is unavailable locally. Existing fitting and annual launchers are unchanged.

Saved-fit Figure 4 reporting (2026-09-28): `python -B -m core.figure4_report
--experiment PATH [--years 1987]` reads annual imports and local Parquet data;
never refits. Separate generic streaming zero-out evaluator supports nonlinear
prediction callbacks and an exact linear shortcut. Current loader supports saved
linear coefficients/intercept/features; existing ENet and OLS-3 have them, older
PCR/PLS imports do not. Future nonlinear model loaders remain to be implemented.
Grouping base characteristics with their macro interactions, and SIC dummies
jointly, is explicitly author-code-unverified. Preserve negative scores; use raw
plots if mean scores cannot be normalized without clipping. Timestamped outputs
go under `results/replication_comparison/<timestamp>/figure4/`. Five new tests and
13 existing evaluation tests passed. Real 1987 ENet check reproduced all 82,945
test predictions within 2.09e-17; training importance covered 475,906 observations
in 1957–1974, taking 54.2 seconds including prediction verification. Full 30-year
importance remains unrun. See `docs/working_notes/Figure 4 Workflow.md`. New
reporting modules are outside fit-resume source hashes; training code is unchanged.

RF pilot (2026-09-28): `run_experiment.py rf --local --sample p05 --years 1987
--cpus 8` uses `gkx_rf_v1`. Paper/appendix settings: 300 bootstrap trees, depth
1–6, squared-error CART. Explicit pilot feature grid completes the appendix's
ellipsis with [3,5,10,20,30,50,100,200,400,920]. Standard scikit-learn stopping,
float32 predictors, float64 targets, seed 2020+year; validation MSE selects one
training-only forest without refit. Figure 3 records average realized tree depth.
Annual checkpoints, reports, launcher and collection support RF. CPU-capped tree
fitting is parallel; candidates and prediction accumulation are serial. See
`docs/working_notes/Random Forest Workflow.md`. No server jobs submitted.
Verification: 192 tests passed, including bootstrap leaf means, tree averaging,
thread-count reproducibility, independent validation-score checks, annual resume,
launcher integration and replication plots. Scikit-learn 1.9.1 installed/tested;
dependency and uv lock updated. The real 5% annual RF run is left to Phil.

GLM+H empirical pilot (2026-09-27): `run_experiment.py glm --local --sample p05
--years 1987 --cpus 8` uses `gkx_glm_huber_v1`. Phil approved training-only spline
centering/scaling, singleton SIC groups, fixed training-mean intercept and squared
penalty selection followed by Huber refinement. Threshold uses original-input
training OLS absolute residual .999 quantile. Objective is mean half-loss plus
unweighted group norms; ten penalties 1e-4..1e-1, an explicit pilot convention.
`core/models/glm.py` / `core/dense_glm_run.py` preserve complete annual arrays,
save transforms/coefficients/convergence, support annual resume and report active
input groups for Figure 3 and zero-benchmark R2 for Table 1. No cluster submission
or real-panel GLM fit performed; Phil will run the 5% check. See
`docs/working_notes/GLM Huber Workflow.md`. Existing remote checkouts must stay
unchanged while jobs need their execution hashes. Progress tracker is Phil-only.

GLM reference comparison (2026-09-27): `python -B -m core.data_examine.glm_matlab
prepare` creates small shared synthetic MAT inputs and Python reference outputs.
The saved `run_glm_reference.m` calls the authors' original Simu_Matlab functions;
`compare --run-dir PATH` checks intermediate transforms, tuning, coefficients,
objectives, and predictions and writes comparison.json/csv. This reproduces
simulation conventions, including separate evaluation centering and summed loss;
it is not an approved empirical GLM specification or integrated production model.
Python numerical checks passed. Phil ran MATLAB R2026b; saved reference_v1
comparison passed all checks, with Huber test predictions within 4.45e-16.
See `docs/working_notes/GLM MATLAB Comparison.md`. Keep existing runs unchanged.

AI Hub GNU time path correction (2026-09-29): the annual, Figure 4 and OLS3
threshold AI Hub shell scripts check and invoke `$HOME/.local/bin/time`, which
Phil verified on a compute node. SCRP scripts retain `/usr/bin/time`.
This local script change does not alter research settings or submit remote jobs.

AI Hub deployment preparation (2026-09-27): `run_experiment.py --aihub` uses
the separate `run_annual_aihub.sh`, backend record and default root
`results/aihub_experiments/`. Use `--aihub` for list/resume/collect ID lookup too.
Explicit root/environment overrides still take precedence. One concurrent task
by default; documented student-budget checks: 240G aggregate array memory,
60 aggregate CPUs, 24h/task. Account/partition/QoS are omitted unless supplied.
SCRP scripts/config/default behavior and all annual worker/model sources remain
unchanged. Preserve the active remote SCRP checkout. New AI Hub jobs must be
prepared on that cluster with matching environment/data; no remote submission or
numerical comparison has been performed. See `docs/Running Python on AI Hub.md`.
Local verification: all 175 unit tests passed, including four new AI Hub launcher
tests. Bash syntax/runtime testing is deferred because Bash is unavailable in
the local environment; Slurm submission and cross-cluster numerical checks remain pending.

ENet+H / OLS-3+H additions (2026-09-26): `run_experiment.py enet/ols3`
uses `gkx_enet_huber_v1` / `gkx_ols3_huber_v1`; sample and years remain CLI choices.
Dense annual fitting is in `core/dense_linear_run.py`, dispatched by the existing
Parquet runner. Original LinearRegressor/APG retained; no batch QR or method change.
ENet uses rho=.5, ten log-spaced penalties 1e-4..1e-1 (explicit pilot choice).
Both use the approved training-only OLS absolute-residual .999 quantile, computed
once per year and fixed across penalty candidates. OLS-3 reads only ranked
mvel1/bm/mom12m. Intercept enabled, no additional scaling, MSE validation, no refit.
Saves selected coefficients/intercept/features, actual threshold, every candidate's
convergence diagnostics, penalty and nonzero slope count. Figure 3 now supports
ENet nonzero slopes; OLS-3 has no tuned-complexity plot. Reports/annual checkpoints
and list/resume/collect share the existing workflow. See the primary run guide and
`docs/working_notes/Huber Linear Workflows.md` for settings and limitations.
Phil reports PCR finished and PLS running on SCRP. Keep that PLS checkout
unchanged while queued/running/resumable jobs need it; source hashes change with
these additions. No remote jobs were submitted or modified during implementation.
Verification: 171 tests passed, including independent SciPy Huber/SLSQP solutions,
feature projection/order, shared training-only delta, staged/complete equivalence,
resume skipping, nonconvergence rejection and launcher/report integration.
Real 1987 fixed-5% launcher runs completed and collected:
`enet_p05_1987-1987_20260926T072037893220Z`: lambda=.01, 11 nonzero slopes,
R2=.523733%, peak Python RAM=.641 GiB, worker elapsed190.15s;
`ols3_p05_1987-1987_20260926T072039517817Z`: R2=.345153%, peak=.140 GiB,
worker elapsed1.74s. These are one-year pilot checks, not full-sample results.

Unified interface: `run_experiment.py pcr/pls --local/--scrp --sample p05/full
--years FIRST:LAST`, plus `list [--refresh]`, `resume ID`, and `collect ID`.
Uses existing baselines PCR=v5 and PLS=p05_v1; writes all resolved/year configs
inside one experiment directory, no ad hoc files in config/. Calls sbatch via
subprocess and records submission IDs/resources/commands in launch.json. Listing
scans these records and separates saved checkpoint counts from optional Slurm
status. Existing main/annual worker/model/launch scripts are unchanged, so current
runs retain their execution identities. Legacy manually launched experiments
are not automatically listed. See `docs/Running Experiments.md` (primary guide)
and `config/README.md`. No cluster jobs were submitted during implementation.
End-to-end local checks through the unified command completed for 1987 on the
fixed 5% sample: PCR R2=0.463759%, PLS R2=0.581663%; both produced combined
reports and discoverable launch records under `results/experiments/`.
Verification: 165 tests passed, including mocked Slurm submission/accounting,
retry protection, preserved settings and listing. Real PCR resume recognized its
completed checkpoint without refitting. Phil confirmed deployment and submission
of full PCR 1987-2016 as Slurm job 397900, experiment
`pcr_full_1987-2016_20260925T182454226202Z`; completion is not yet verified.
`docs/Running Experiments.md` now starts with a daily command reference using
that real job: status/accounting, job-to-experiment lookup, logs, peak RAM,
collection, retries and cancellation. Keep this as the primary run guide.

Independent annual tasks (2026-09-26): `core.annual_jobs prepare/run/collect`
and `run_annual_scrp.sh` reuse main/model/loading code unchanged. Preparation
snapshots one config per test year and separate run parents; retries resume the
same annual run. Default array 0-7%8 requests 8 CPUs/128G/two hours per task.
First intended SCRP experiment: PCR 2009-2016. Collection checks complete,
compatible, nonduplicate annual checkpoints and creates pooled Table 1 and
Figure 3 diagnostics without loading features. See
`docs/working_notes/Annual Parallel Runs.md` for commands and limits.
Phil's standalone 2016 SCRP result: K=6, R2=0.709179%, peak RSS 96.275 GiB,
elapsed 730.68 seconds. This supersedes the earlier unmeasured full-window
estimates below. Eight simultaneous tasks have not yet been tested on SCRP.
Local verification: 155 tests pass. Two real 5% annual tasks (1987/1988) and
collection completed in `results/pcr_parallel_local_check_20260926`; K=14/16,
pooled R2=0.477986%, peaks 0.980/1.100 GiB. Selected K and predictions agree
with the saved v1 pilot (absolute prediction tolerance 1e-10). Bash syntax
check passed. `gkx_pcr_p05_check` is the bounded local two-year config.

Memory counters now auto-select Windows APIs or Linux /proc + getrusage;
`main.py --memory` and the existing memory diagnostics use the same function.
Linux private committed memory is unavailable (null), not substituted by virtual
memory. Peaks cover the current process only, excluding child processes.
For 2016, `core.data_examine.pcr_full_memory --test-year 2016 --components 100`
is an existing fixed-K diagnostic. The tuned `gkx_pcr_2016_check` config was
provided as a server-side creation command and is not present locally.
See `docs/Running Python on SCRP.md` for both commands. No new config was added.

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
