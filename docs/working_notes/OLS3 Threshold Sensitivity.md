# OLS-3 Huber threshold sensitivity

This separate diagnostic keeps the production Huber formula, estimator, configs,
annual runners and old results unchanged. It fits four predefined variants:

| Variant | Training-only rule |
| --- | --- |
| ols_residual_q999 | Quantile .999 of absolute OLS-3 residuals (current baseline) |
| absolute_return_q999 | Quantile .999 of absolute excess returns |
| demeaned_return_q999 | Quantile .999 of absolute excess returns minus their training mean |
| ols | Ordinary squared-loss OLS-3 reference |

The third rule means `quantile(abs(y - mean(y)), .999)`. All quantiles use NumPy
linear interpolation. The paper's equation (6) defines the loss and appendix
Table A.5 specifies a 99.9% quantile, but the empirical quantile construction is
unresolved. These alternatives are sensitivity checks, not recovered author rules.
No threshold is selected by validation or resemblance to the paper's test R2.

Features are mvel1, bm, mom12m; fitted intercept, no additional standardization,
no penalty, APG max_iter=50000, tol=1e-8, initial_step=1. Training begins in 1957;
for test year Y, training ends Y-13, validation spans Y-12 to Y-1, and testing is Y.
No validation refit. Only three predictor columns are loaded, once per window,
and shared by all variants. The original Parquet reader verifies source checksums;
this can dominate runtime because the source files contain all 920 features.
Nonconvergence fails the experiment rather than silently reporting a bad fit.

## Run on AI Hub

Upload the new `core/data_examine/ols3_thresholds.py` and
`run_ols3_thresholds_aihub.sh` into the same paths in your existing AI Hub checkout.
Use its Linux `.venv/bin/python` and prepared `data/processed/gkx_parquet` (or
`gkx_dev_p05_seed2020_parquet` for p05). Do not upload the Windows virtual environment.
Existing model/data-reader modules and dependencies must match this checkout.
See [AI Hub setup](../Running%20Python%20on%20AI%20Hub.md) for initial deployment.

From the AI Hub project root:

```bash
# Lightweight syntax/import check, with no fitting:
bash -n run_ols3_thresholds_aihub.sh
.venv/bin/python -B -m core.data_examine.ols3_thresholds --help

# Four full-universe test years in one CPU job:
sbatch run_ols3_thresholds_aihub.sh --sample full --years 1987 1988 2011 2012

# Later, all 30 test years in a new timestamped run:
sbatch run_ols3_thresholds_aihub.sh --sample full --years 1987:2016

# Optional small-sample check instead:
sbatch run_ols3_thresholds_aihub.sh --sample p05 --years 1987
```

Defaults request 4 CPUs, 8G RAM, two hours, no GPUs, and no array. These are initial
allocations, not measured AI Hub requirements. Account/partition/QoS use site
defaults. If your account needs explicit settings, put confirmed flags **before**
the script name: `sbatch --partition=YOUR_CPU_PARTITION ... run_ols3_thresholds_aihub.sh ...`.
Inspect `sinfo`, `scontrol show partition`, and account associations as described
in the setup guide; do not assume a partition name. The
[official guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide)
documents Slurm submission and installation before compute-node execution.
No live quotas or cluster execution were checked by this addition.

This standalone diagnostic uses direct sbatch and is not registered with
`run_experiment.py list/resume/collect`. To monitor, substitute the submitted ID:

```bash
squeue -j JOBID
cat slurm-ols3-thresholds-JOBID.out
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

The Slurm output points to `results/huber_threshold_logs/aihub-JOBID-TIMESTAMP/run.log`.
That log prints the exact experiment output directory. Neighboring `time.txt`
records GNU time resource use; package versions, BLAS configuration and submitted
script are saved too. Run locally with `python -B -m core.data_examine.ols3_thresholds`
and the same options, without sbatch.

## Outputs

Each invocation creates `results/huber_threshold_sensitivity/<UTC timestamp>/`:

- `manifest.json`: settings, exact years, data identity, source hashes, versions,
  completed years and running/complete/failed status.
- `table1/comparison.csv` and `.png`: All-stock pooled test R2 for each variant,
  pooled MSE, and difference from the original threshold in percentage points.
  Pool squared-error sums; do not average annual R2. A four-year result is not
  directly comparable with the paper's 30-year number.
- `diagnostics/annual.csv`: annual thresholds, fitted training tail fractions,
  coefficients/intercept, optimizer diagnostics, validation MSE, annual test R2,
  and prediction differences from the baseline. PNG plots show thresholds,
  tail fractions and annual test R2. These are diagnostics, not paper Figure 3.
- `annual/Y/fits.json`: complete annual numeric records for all variants.
- `annual/Y/{validation,test}_predictions.npz`: source row positions, return months,
  observed returns, and one prediction array per variant. No pickle required.

Raw training predictions are not saved; training MSE and tail fractions are.
Coefficients allow their reconstruction. Huber objective values at different
thresholds are not comparable performance scores. All-stock only, no size subsets.
Old outputs are never overwritten. There is no resume command: failed runs retain
finished annual files and status; rerun remaining years in a fresh directory.
Combined reports are written only after every requested year succeeds. Copy the
timestamped output folder back with SFTP to inspect it locally.

## Verification (2026-09-28)

Three focused tests passed: equivalence with the existing baseline fitting path,
training-only threshold construction, saved-prediction pooled R2 accounting,
annual windows, and failure status. A real 1987 fixed-5% run completed at
`results/huber_threshold_sensitivity/20260928T151415449711Z`; the comparison PNG
was visually inspected. Test R2 (%): baseline 0.345153, absolute-return threshold
0.345669, demeaned-return threshold 0.345332, ordinary OLS 0.331997. The three
Huber fits each converged in 26 iterations. These are one-year small-sample
checks, not full-sample findings. Full-sample and AI Hub execution remain unrun.
Bash is unavailable locally; run the documented `bash -n` check on AI Hub.
