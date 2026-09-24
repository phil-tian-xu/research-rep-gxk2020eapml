# Annual fitting from Parquet batches

`gkx_table1_figure3_v4` uses the existing full-universe `gkx_parquet` conversion,
`DATA.FORMAT: parquet`, and `DATA.BATCH_SIZE: 10000`. The CSV and prepared values
are unchanged. The original CSV backend remains available for other configs.

## Run commands

Start a **new** run because the backend and execution source hashes changed:

```powershell
# Optional first-year check, then resume this same directory for the first stage.
python -B main.py --config gkx_table1_figure3_v4 --stop-after-year 1987
$runDir = 'results/gkx_table1_figure3_v4/REPLACE_WITH_PRINTED_RUN_ID'
python -B main.py --resume "$runDir" --stop-after-year 1996
python -B main.py --resume "$runDir" --stop-after-year 2006
python -B main.py --resume "$runDir" --stop-after-year 2016
```

To start directly with 1987–1996, use `--stop-after-year 1996` in the first command.
Omitting the stopping year runs all remaining configured years. Incomplete runs
print the exact directory and resume command. Interrupting a fit retains earlier
completed model-years; the incomplete model restarts on resume. The small QR
summary is recomputed for a partially completed year and released between years.

## Data selection and numerical methods

The first split is unchanged: training return years 1957–1974, validation
1975–1986, test 1987. Training expands yearly, validation rolls by one year,
and each test window is one year. Every fit uses all observations in its
training window. There is no separate model per batch and no validation refit.

`ParquetPanel` selects manifest parts by return year and iterates Arrow batches
with column projection and threaded read-ahead disabled. It verifies selected
parts' SHA-256 once per invocation, plus row counts, float64 schemas, finite
values, month alignment, and original row positions. It checks file size/mtime
again on later reads. The manifest hash covers the entire ordered part list.
Unused years are neither decoded nor hashed. This trusts the completed converter
for source asset/month uniqueness; it does not rebuild an all-panel identifier
index. Completed test predictions are checked against projected batch reads on
resume. Dataset/config/source/package identity checks and run locking still apply.

Training first computes means and optional population SD using only training
rows. Constant columns and constant targets retain exact offsets. A second pass
performs blocked QR on `[centered/scaled X, centered y]`, combining each batch
with the previous small triangular factor. At most a batch plus a roughly
921-by-921 factor is reduced at once. It never forms the full design, full
latent score matrix, or normal equations X'X (which square the condition number).

PCR uses one SVD of the reduced design for the entire component grid. Its
numerical rank threshold uses the **original** training dimensions. PLS applies
the existing SIMPLS routine to the QR reduction: orthogonal transformation
preserves the covariances, score inner products, and final least squares.
Coefficients are mapped back to original feature units before prediction.
K=0 still means the training mean when the intercept is enabled.

OLS+H uses the same unpenalized Huber objective and `svd_apg` preconditioning.
Each objective/gradient evaluation sums observation-weighted contributions
from **all training batches**, then the optimizer makes one global update.
It is not stochastic optimization. The threshold remains the exact 99.9th
percentile of absolute training OLS residuals, using one float64 residual per
training row. A zero threshold still fails explicitly. Whitened and original
gradient convergence checks remain in place; nonconverged fits are rejected.

The numerical coordinate changes can cause small floating-point differences
from a single dense decomposition. Rank boundaries, nearly tied validation
losses, and ill-conditioned models deserve special scrutiny. No regularization,
prediction clipping, changed grids, subsampling, or float32 conversion was added.
The residual-quantile construction remains our previously approved interpretation
of the paper, not a newly verified author rule.

## Memory, output, and speed

A 10,000-by-920 float64 feature batch is 70.19 MiB. Arrow decoding, NumPy copies,
QR work arrays, candidate coefficients, and BLAS libraries add overhead.
The main fitting memory scales with batch_rows × features and features²,
rather than training_rows × features. The Huber quantile vector needs 8 bytes
per training row: at most 28.54 MiB even if it covered all 3,740,946 panel rows.

Validation scores are accumulated over batches; only batch_rows × candidate_count
forecasts are retained. Each completed model's annual prediction columns are
saved atomically in the existing model-year checkpoint format. These small
annual prediction tables are retained for serialization, not the year's features.
Aggregate CSV and Table 1 reports pool sums and read checkpoints one model-year
at a time. Figure 3 uses the saved tuning records. The batch API returns the
prediction CSV path instead of loading all predictions into its result object.

Huber repeatedly decompresses training parts, and QR adds computation; this
implementation addresses RAM use, not maximum throughput. Progress messages
identify the year, model, QR progress, and Huber passes. Checksums also add one
disk pass per selected part per invocation. No cache requiring tens of GB of
RAM or extra full-panel scratch files is created.

## Verification

For the next real-data checks (saved 5% results, then first/largest full-universe
windows against an 8-GiB peak resident target), use the standalone script and
commands in [Batch Verification](Batch%20Verification.md). The synthetic tests
and bounded smoke measurement below do not replace those checks.

All 126 repository tests pass (including nine new batch-workflow tests).
Automated checks compare dense versus batched PCR/PLS predictions across batch
sizes and centering/scaling choices, and dense versus batched Huber objectives,
thresholds, convergence, and predictions. Cases include duplicate/constant
features, zero components, constant targets, more columns than rows, and the
original-dimension numerical rank threshold. End-to-end tests compare selected
parameters, scores, and predictions with the dense runner, then compare staged
and uninterrupted batch runs. They cover model-year failure recovery, main.py
dispatch, projected/year-filtered reads, corruption checks, and lazy reports.

A bounded real-data check used all 10,671 rows in return year 1957 (920 features),
PCR K=100, PLS K=10, and configured OLS+H; each predicted 10,000 real 1987 rows.
All forecasts were finite and Huber converged in 8 iterations. Training rank was
224 and Huber delta was 0.388824538. Peak Windows process resident RAM was
0.563 GiB, with private committed memory around 1.745 GiB at the end; elapsed
time was about 33 seconds. These are different memory measures, not additive.
This is a technical smoke test, **not** the actual 1987 training window or a
replication result. A complete full-universe annual/30-year run has not been
executed in this verification; the measurement is not an unconditional 64-GB
machine-wide memory guarantee. The pilot's Huber empirical instability remains
an open research issue.
