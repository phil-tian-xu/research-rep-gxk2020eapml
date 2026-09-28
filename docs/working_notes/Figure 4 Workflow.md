# Figure 4 from saved fits

This is a separate reporting workflow. It never trains models or repeats tuning.
The current ENet imports contain the required state for all 30 years. Existing
PCR/PLS imports lack coefficients and must be skipped. OLS-3 can be reported as a
supplementary three-characteristic diagnostic, not an original Figure 4 panel.

From the repository root, start with one test year:

```powershell
.venv/Scripts/python.exe -B -m core.figure4_report --experiment results/scrp_imports/enet_full_1987-2016_20260926T080544288119Z --years 1987
```

Omit `--years 1987` for 1987–2016. `--years FIRST:LAST` selects an inclusive
range. Supply multiple experiment paths after `--experiment` to report more
models with saved fitted state. Each experiment must have the annual-run folder
layout. Ambiguous/missing annual runs and duplicate models are rejected.
`--data-dir`, `--output-root`, and `--batch-size` optionally override defaults.

Each invocation creates `results/replication_comparison/<timestamp>/figure4/`:

- `<model>.png` and `.svg`: top 20 groups, or all groups when fewer than 20 exist.
- `annual_importance.csv`: signed annual importance and SSE increases for all groups.
- `mean_importance.csv`: equal-weight annual means, relative shares when valid,
  and normalization status. Normalization includes all groups, before top-20 selection.
- `verification.csv`: restored test-prediction discrepancies, training dates,
  sample counts, benchmark SSE, baseline SSE/R2 and elapsed seconds.
- `methodology.json`: generation time, sources, dataset identity, groups and choices.
  Status remains `incomplete` after an interruption/failure; annual CSVs retain
  completed years. This initial version does not resume reporting runs.

## Run on SCRP or AI Hub

Use `run_figure4_scrp.sh` or `run_figure4_aihub.sh` from the respective Linux
project root. Both request one node, one process, 4 CPUs, 8G RAM and four hours;
these are initial allocations, not measured cluster requirements. No GPUs or
arrays are requested. The selected years run sequentially and produce one
combined report, avoiding a separate year-collection step.

The scripts follow the [official SCRP Slurm guide](https://scrp.econ.cuhk.edu.hk/guide/slurm)
and [AI Hub guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide),
checked 2026-09-28. SCRP defaults match the recorded account: `pg`, partition
`scrp`, QoS `c32g4`. AI Hub leaves these three settings unspecified; supply
confirmed site options to `sbatch` if required. Current account eligibility and
availability have not been queried. On SCRP, inspect `qos` and `scrp-info`; on
AI Hub, inspect `sinfo` and your account association as described in the AI Hub run guide.

Deploy the two new Python modules (`core/figure4_report.py` and
`core/figure4_importance.py`) and the appropriate shell script. Do not overwrite
training sources or environments needed by active model jobs. Preserve LF line
endings for the shell scripts (`.gitattributes` already specifies this).
The cluster environment needs the existing project dependencies and GNU
`/usr/bin/time`. Use the Linux `.venv/bin/python`, not a copied Windows environment.

SCRP can read the original completed experiment directly; it does not need a
copy under `scrp_imports`. For AI Hub, copy that completed experiment's `annual/years/`
tree, including manifests, resolved configs and checkpoint JSON files, plus the
unchanged prepared Parquet dataset and metadata. Combined Table 1/Figure 3 plots
are not needed. Remote absolute output paths embedded in old configs are not
used for this read-only reporting job. Data identities and restored predictions
are checked on the destination host.

### Submit

Run in Bash on SCRP (no preceding `compute` allocation is needed):

```bash
cd ~/research-rep-gxk2020eapml
sbatch run_figure4_scrp.sh \
  --experiment /data/users/s1155269217/GKX2020/results/experiments/enet_full_1987-2016_20260926T080544288119Z \
  --data-dir /data/users/s1155269217/GKX2020/data/processed/gkx_parquet \
  --output-root /data/users/s1155269217/GKX2020/results/replication_comparison
```

On AI Hub, from its project root, with the completed experiment copied into the
same relative import layout as the local machine:

```bash
sbatch run_figure4_aihub.sh \
  --experiment results/scrp_imports/enet_full_1987-2016_20260926T080544288119Z \
  --data-dir data/processed/gkx_parquet \
  --output-root results/replication_comparison
```

Both commands default to all 30 years. Add `--years 1987` for a one-year check,
or `--years 1987:1996` for a partial-period diagnostic. Additional supported
model experiment paths can follow `--experiment`; scripts are not ENet-specific.
Model-loading support is still required, and old PCR/PLS imports remain unsupported.

Place resource overrides **before** the script name; report options go after it:

```bash
sbatch --cpus-per-task=4 --mem=16G --time=08:00:00 run_figure4_scrp.sh \
  --experiment /absolute/path/to/completed/experiment
```

### Inspect and retrieve

Slurm prints a job ID. `slurm-figure4-JOBID.out` points to the detailed log under
`results/figure4_logs/<site>-<jobid>-<UTC timestamp>/` in the submission checkout.
That directory contains `run.log`, `time.txt` (GNU time including peak RSS),
package versions, NumPy configuration, the submitted script and report-source hashes.
BLAS threads are capped to allocated CPUs; logs use unbuffered UTF-8 output.

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
cat slurm-figure4-JOBID.out
# Use the exact path printed above:
tail -f results/figure4_logs/SITE-JOBID-TIMESTAMP/run.log
```

The final `Saved Figure 4 diagnostics:` log line identifies the timestamped
result directory under the requested output root. Download that whole directory
using SFTP; it includes the PNG/SVG, CSVs and methodology record. Confirm
`methodology.json` says `complete`. A one-year report is not the full Figure 4.

Cancellation uses `scancel JOBID`. Reporting does not yet support resume: retrying
creates a new timestamped report and recomputes the requested years. Existing
partial outputs and fitted checkpoints are preserved. No cluster jobs have been
submitted or validated by this addition. Bash execution is untested locally
because this Windows environment has no Bash executable.

## Definition and explicit choices

Paper sections 1.9 and 2.3, printed pages 2247 and 2254–2255, define importance
as the reduction in training-sample predictive R2 when a predictor is zeroed
while holding fitted parameters fixed. Average over the annual training samples,
then normalize within the model. This uses squared errors even for Huber fits.

For each annual fit, the saved config determines the original expanding training
window: FIRST_TRAIN_YEAR through test_year - VALIDATION_YEARS - 1. Each window
gets equal weight in the final mean, regardless of its observation count.

The current grouping is an explicit, **author-code-unverified** convention:
jointly zero a base characteristic and every macro interaction containing it;
jointly zero all SIC dummy columns as one `sic2` group. The normalizing universe
includes industry. Do not interpret this as a fully verified reproduction of
the authors' feature aggregation. Grouping is recorded and stated on plots.

Negative annual values are retained. If any *mean* importance is negative, or
the sum is zero, relative shares remain missing and the figure plots raw signed
R2 reductions. No absolute values or clipping are applied. A zero-coefficient
annual model has zero importance; retain that year when averaging.

## Verification and efficiency

The report checks checkpoint SHA-256, saved config consistency and dataset identity.
The Parquet reader verifies the parts it reads. All test observations, months,
returns and restored predictions must match the checkpoint before computing an
annual importance score. Prediction tolerance is atol=rtol=1e-10; observed return
tolerance is atol=1e-14, rtol=0. Training row counts must also match the saved fit.
These checks do not certify that original estimation assumptions match the paper.

The evaluator consumes batches, with no full training matrix allocation. For a
linear model, subtract the group's contribution from the baseline prediction.
If e = y - prediction and c is that contribution, the SSE increase is
sum(c*c + 2*e*c). This is exactly the fixed-model zero-out calculation.
Zero coefficients can be skipped without changing the result.

Runtime includes reading and hashing data. Thirty overlapping training windows
still require substantial I/O; a one-year elapsed time is not a guaranteed full-run
estimate. Plotting itself is inexpensive.

## Adding models later

`core/figure4_importance.py` accepts any row-independent `predict(X)`
and a stream of arrays. Linear coefficients optionally enable the exact shortcut;
nonlinear models use direct zeroed-input predictions. No model classes or fitting
code are required by the evaluator.

`restore_predictor` in `core/figure4_report.py` is the small model-loading extension
point. Add a loader for an actual saved nonlinear representation when it exists,
returning `(predict, feature_names, None)`. The callback must apply the original
saved preprocessing. Never treat GLM spline coefficients as raw-feature slopes.
Future PCR/PLS coefficients must be in original input coordinates with an intercept
and exact feature order; existing model classes provide these, but older runs did
not save them. Current recognized linear loaders reject missing fitted state.
No training or checkpoint writer was changed by this reporting addition. The new
modules sit outside the directories automatically included in fit-resume hashes.

Focused checks:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -p test_figure4_report.py -v
```

Checks cover literal versus shortcut importance, batching, nonlinear callbacks,
group validation, zero targets, missing fitted state, signed values, averaging
before normalization, and a synthetic saved-checkpoint-to-plot workflow with a
deliberately corrupted fitted predictor rejected.

Real-data check (2026-09-28): the saved 1987 ENet fit reproduced all 82,945 test
predictions within 2.09e-17 maximum absolute error. Its training importance used
475,906 observations from 1957–1974, with verification plus calculation taking
54.2 seconds. Outputs are under
`results/replication_comparison/20260928T171334534660+0800/figure4/`.
This is a one-year diagnostic; the full 30-year importance calculation has not run.
