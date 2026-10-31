# Neural networks: specification and implementation

This is a PyTorch implementation of Gu, Kelly and Xiu (2020), *Empirical Asset
Pricing via Machine Learning*, Section 1.7, Internet Appendix B.3 and Table A.5.
It uses this repository's prepared panel and existing annual prediction protocol.
It is a documented implementation of the published method; the authors' exact
empirical NN training code and every numerical convention have not been recovered.

## Where to review or reuse the model

- `core/models/neural_network.py`: architecture, objective, initialization,
  optimization, stopping, prediction and portable ensemble state. Receives arrays
  and ordinary arguments; has no project configuration or Parquet dependency.
- `config/gkx_nn1_v1.yaml`: complete baseline tuning/training settings.
  `gkx_nn2_v1.yaml` through `gkx_nn5_v1.yaml` inherit it and change architecture.
- `core/nn_run.py`: annual data windows, candidate selection, baselines and outputs.
- `tests/test_neural_network.py`: independent objective/gradient, known-signal
  learning, deterministic reruns, stopping/buffer restoration, signed output,
  serialization and CUDA-to-CPU prediction checks.
- `tests/test_nn_workflow.py`: annual selection, prediction restoration and resume.

## Published settings and explicit implementation choices

| Item | Baseline | Provenance |
| --- | --- | --- |
| NN1 hidden widths | 32 | Paper Section 1.7 |
| NN2 | 32, 16 | Paper |
| NN3 | 32, 16, 8 | Paper |
| NN4 | 32, 16, 8, 4 | Paper |
| NN5 | 32, 16, 8, 4, 2 | Paper |
| Hidden blocks | Fully connected -> ReLU -> batch normalization | Appendix B.3 explicitly places BN after ReLU |
| Output | One unrestricted linear scalar, with bias | Paper Equation (18) |
| Training error | Mean half squared error in supplied return units | Explicit normalization of the paper's squared-error objective |
| Penalty | L1 sum over every dense kernel, including output | L1 is published; precise penalized parameter set is our choice |
| Learning-rate candidates | 0.001, 0.01 | Table A.5 |
| L1 candidates | 0.00001, 0.0001, 0.001 | Our three-point log discretization of the published range |
| Batch size | 10,000 observations | Table A.5 |
| Maximum epochs | 100 | Table A.5 |
| Early-stopping patience | Five consecutive epochs without a strictly lower validation MSE | Table A.5 / Algorithm 6; monitoring once per epoch is explicit |
| Ensemble | Equal average of ten independently initialized networks | Table A.5 and paper |
| Adam | beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=0, amsgrad=False | Explicit Adam defaults; foreach/fused disabled for a simple deterministic path |
| Dense initialization | Glorot/Xavier uniform kernels; zero biases | Our choice; the paper does not specify the initialization distribution |
| BN | epsilon=1e-5, momentum=.1, gamma=1, beta=0, running statistics | Explicit PyTorch conventions; exact empirical author settings unavailable |
| Additional predictor scaling | None; consume the existing prepared features | Our baseline choice; preserves ranked characteristics and supplied macro interactions |
| Target scaling | None; decimal excess returns throughout | Repository's data convention; never convert to percent inside the trainer |
| Numerical precision | float32 training/inference; float64 MSE scoring and ensemble accumulation | Our implementation choice |
| GPU path | Single GPU, eager operations, deterministic algorithms; TF32 disabled | Our reproducibility choice |

No Huber modification, dropout, extra L2/Adam weight decay, learning-rate
scheduler, gradient clipping, target winsorization, automatic target scaling,
mixed precision, monthly refitting, or warm starts are added. Adam's adaptive
steps implement the optimizer described in the appendix; no extra decay schedule
is inferred from the paper's phrase "learning rate shrinkage."

## NN6-NN10 research extensions (2026-10-31)

Phil approved deeper, wider pyramids extending the existing halving pattern. These are our research extensions; the paper specifies NN1-NN5. The original NN1-NN5 widths remain unchanged.

| Model | Hidden-layer widths | Config |
| --- | --- | --- |
| NN6 | 64, 32, 16, 8, 4, 2 | `gkx_nn6_v1` |
| NN7 | 128, 64, 32, 16, 8, 4, 2 | `gkx_nn7_v1` |
| NN8 | 256, 128, 64, 32, 16, 8, 4, 2 | `gkx_nn8_v1` |
| NN9 | 512, 256, 128, 64, 32, 16, 8, 4, 2 | `gkx_nn9_v1` |
| NN10 | 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2 | `gkx_nn10_v1` |

`core/models/nn_architectures.py` records all ten architectures and supplies model recognition to the launcher, annual preparation, dense dispatch, reports and saved-fit restoration. It has no PyTorch dependency. Importing `ARCHITECTURES` from `core.models.neural_network` remains supported. The new configs inherit `gkx_nn1_v1`; they change only the enabled model, config name and default output directory. Training remains mean half-squared error plus summed dense-kernel L1, with the same initialization, Adam settings, validation selection, ten-member ensemble, stopping, splits and numerical precision. There is no Huber-versus-MSE change between feature specifications.

Depth and width both increase in these extensions, so their comparison does not isolate depth alone. The L1 sum is still unnormalized by parameter count; the same candidate grid can imply different effective regularization for larger networks. Review validation curves and selected penalties. This extension supplies no evidence yet that larger networks improve predictive performance. Keep the previous NN1-NN5 results and evaluate the extensions under the recorded research protocol.

Use `run_experiment.py nn6` through `nn10` with the same `--dataset`, `--config`, `--local`/`--scrp`/`--aihub`, `status`, `resume` and `collect` interface. Complete saved fits support Figures 4/5 through the existing restoration path and NN prediction tolerances. The architecture module is included automatically in runtime source hashing. Adding these models changes execution fingerprints: prepare fresh experiment IDs and retain the original checkout/environment for historical resumes; reporting on saved results remains available.

## Exact loss and units

For a training mini-batch B:

```text
data_loss = (1 / (2 * len(B))) * sum((y - prediction)**2)
penalty   = lambda * sum(abs(W)) over dense-layer kernels W
objective = data_loss + penalty
```

The penalty is a **sum**, with no division by parameter count or batch size.
Biases and BN gamma/beta are excluded. This exclusion is a documented choice,
not a claim about the unavailable empirical code. `Adam.weight_decay` is zero:
weight decay would not implement this L1 penalty.

Pooled stock-month rows receive equal weight, matching the current workflow's
MSE. Appendix B.3 also writes its objective using time and cross-sectional
averages; the unbalanced-panel weighting of the empirical NN implementation is
not verified. We do not silently introduce equal-month or market-cap weights.
Changing weights, half-MSE versus MSE, return units, or summed versus averaged
penalties changes the effective regularization. Such changes require a versioned
specification and a new experiment, even if the numerical lambda grid is unchanged.

Predictions and targets must both have shape `(batch,)`. The loss explicitly
rejects a `(batch,1)` versus `(batch,)` mismatch to prevent accidental broadcasting.
Negative return forecasts are allowed; the output has no ReLU or sigmoid.

## Batching, normalization and stopping

Each epoch draws a new permutation of **training rows only**, then visits every
row exactly once, without replacement. Shuffling does not change the chronological
train/validation/test assignment. A final singleton is merged into the preceding
batch because BN cannot estimate variance from one row; no observation is dropped.
The final batch may therefore contain 10,001 observations in that edge case.

BN updates its moments only in training mode. Training diagnostics, validation
and test prediction use evaluation mode and frozen training running statistics.
PyTorch running-variance updates use its documented unbiased variance convention,
while the training forward pass uses batch variance. This can differ from older
framework conventions even with apparently matching settings.

After each epoch, compute unpenalized validation MSE. Strict improvement resets
patience to zero and saves all weights **and BN buffers**. Stop after five
consecutive non-improvements or 100 epochs, then restore the best saved epoch.
The initial untrained model is logged for diagnosis but is not a selectable
epoch, following Algorithm 6's first validation check after training updates.
Reaching 100 epochs is reported as `max_epochs`, not claimed numerical convergence.

For each learning-rate/L1 candidate, train all ten seeds and evaluate the mean
of their forecasts. Select the candidate with the lowest **ensemble** validation
MSE; exact ties retain the first configured candidate. Common hyperparameters
across members and ensemble-based candidate selection are explicit choices where
the paper does not completely specify the selection order. Do not choose the
best-performing seed or architecture using test returns. NN1-NN5 are separate
reported specifications, not a test-selected architecture contest.

No combined training/validation refit follows selection. The held-out test year
is loaded only after tuning. The annual baseline for 1987 is training 1957-1974,
validation 1975-1986, test 1987; subsequent years expand training and roll the
12-year validation window. Dates label realized return months.

## Seeds, devices and persistence

Member seeds are `base_seed + 1000 * test_year + member_index`, starting at zero.
The baseline base seed is 2020. Every candidate uses the same seed list, with a
separate CPU generator for its training-row permutations. Architectures also use
the same seed list; different widths naturally consume random numbers differently.

CUDA requests fail clearly if unavailable; there is no silent CPU fallback.
Deterministic operations, CUBLAS_WORKSPACE_CONFIG=:4096:8 and disabled TF32 reduce
repeat-run variability on the same environment. Bitwise identity across CPU/GPU,
different GPU models, PyTorch releases or CUDA libraries is not promised.
Record PyTorch/CUDA versions and device name with each fitted year. Selected
weights and BN buffers are saved as numeric JSON arrays, with the ordered feature
list and architecture, inside the existing checksummed annual checkpoint.
Restoration does not execute pickle or train a replacement model.

The Figure 4 loader restores the ensemble on CPU and verifies its saved test
predictions before computing importance. For float32 NN cross-device restoration
the check uses atol=1e-6 and rtol=1e-5 in decimal-return units; other models retain
their existing tolerances. The report saves the actual discrepancy. Changing
device arithmetic can produce small errors; these tolerances are not an excuse
for a materially different forecast.

Resume operates at the completed model-year level. An interrupted unfinished
year restarts its candidate/seed fits; mid-epoch optimizer resume is not provided.
Source and environment checks include the NN runner and installed torch version.
Existing runs need their original checkout/environment; never update an active
remote checkout to resume an older experiment with changed execution hashes.

## Diagnosing near-zero fits

Every candidate/member records initial errors, per-epoch train and validation
MSE, online half-MSE and L1 contribution, first-batch gradient norm, forecast
dispersion, best epoch, stopping reason, parameter change from initialization,
and nonzero dense-weight count. These distinguish weak predictive signal from
frozen updates, an excessive penalty, numerical failure or displayed rounding.

The baseline does not discard a constant predictor just because it is unattractive:
it may be the selected regularized fit. Investigate it using training diagnostics
and an explicitly separate unpenalized diagnostic, not by retuning on test results.
Meaningful synthetic tests establish that the code can learn a known relationship.

## Running the first fixed-5% pilot

Install the optional dependency into the existing uv environment:

```powershell
uv sync --extra nn
.venv\Scripts\python.exe -B run_experiment.py nn1 --local --sample p05 --years 1987 --cpus 4
```

`torch==2.13.0` is pinned; uv uses the official CUDA 13.0 index on Windows/Linux.
The same optional extra supports CPU tests. A different cluster driver/runtime
may require a separately pinned compatible wheel; verify an actual GPU step on
the compute node before a full experiment. Install dependencies before submission;
AI Hub compute nodes have no internet. Keep optional NN dependencies installed
when resuming NN experiments (`uv sync` without the extra can remove them).

`main.py --config gkx_nn1_v1 --stop-after-year 1987 --memory` also works. The config
defaults to the 5% sample and the complete 1987-2016 horizon; the launcher command
above requests exactly one year. The pilot uses the full stated grid and all ten
seeds, rather than quietly shortening the published epoch budget or ensemble.

After a successful local pilot and a matching small cluster check, full runs use:

```bash
.venv/bin/python -B run_experiment.py nn1 --scrp --sample full --years 1987:2016 --parallel 1 --gpus 1
.venv/bin/python -B run_experiment.py nn1 --aihub --sample full --years 1987:2016 --parallel 1 --gpus 1
```

Choose one backend on that cluster. The launcher requests one GPU per annual
task (SCRP defaults to `--gpus=a100:1`, AI Hub to `--gres=gpu:1`); default NN concurrency is one,
initial RAM request 64G and wall time 12 hours. These are initial allocations,
not measured full-sample requirements or promises of current account availability.
Confirm live account/partition/GPU limits before submitting. Existing CPU model
commands never request GPUs. Seeds/candidates run serially on each task's GPU.

SCRP NN submissions select `run_nn_scrp.sh`. `--gpu-type a100` (default) uses
partition `a100` and the documented `a100` node constraint. `--gpu-type rtx3090`,
`rtx3060`, or `any` defaults to partition `scrp` and removes the A100 constraint.
The launcher puts these settings in the submission command; the generic script
has no hidden GPU type or partition. The old A100 script remains for legacy use.
Startup logs the actual GPU and driver
and verifies a CUDA matrix forward/backward pass before the annual worker.
For the first 1987 fixed-5% cluster check, use the smaller resource request:

```bash
.venv/bin/python -B run_experiment.py nn1 --scrp --sample p05 --years 1987 \
  --parallel 1 --cpus 4 --mem 8G --time 00:30:00
```

Run from the updated SCRP project root after installing `uv sync --extra nn`.
The launcher snapshots this batch script and submits it; no manual `sbatch`
command is needed. This pilot command requests one GPU through the NN defaults.
Its CPU/RAM/time overrides do not change model settings or the sample. The local
preparation and mocked submission checks do not establish cluster runtime success.

For the full universe and 1987-2016 with four concurrent RTX 3090 tasks:

```bash
.venv/bin/python -B run_experiment.py nn1 --scrp \
  --sample full --years 1987:2016 --gpu-type rtx3090 --gpus 1 \
  --parallel 4 --cpus 8 --mem 96G --time 12:00:00
```

GPU count is one per annual task. `--parallel 4` allows four independent annual
tasks, requesting up to four GPUs, 32 CPUs and 384G CPU RAM in total, subject to
live account/partition limits and availability. The time limit is per year.
This changes scheduling only: sample selection, loss, precision, batches, seeds,
tuning, ensemble size and early stopping remain identical. Full-sample RAM and
runtime for these settings have not yet been measured.

GPU type and concurrency are saved in `launch.json` and retained on resume.
When explicitly changing GPU type on resume, its default partition replaces the
previous partition unless `--partition` is supplied too. Legacy SCRP NN records
without the new GPU-type field retain the A100 default. AI Hub keeps its untyped
GPU default; explicit types use `--gres=gpu:TYPE:1` with site-specific labels.

Use the usual `list`, `resume`, and `collect` commands. Each annual run contains
the authoritative checkpoint, an additional `NN1_training.csv` covering all
candidates/seeds, and `NN1_training.png` for the selected candidate. The pooled
Table 1 evaluation is reused. There is no Figure 3 complexity curve for these
fixed architectures, consistent with the paper's explanation.

## Combined runs and training diagnostics (2026-10-25)

Disjoint NN year ranges can share one combined ID while retaining separate GPU
requests. See [the run guide](../Running%20Experiments.md#split-gpus-one-experiment-id)
for preparation, submission, status, resume, collection and saved-fit reporting.
The grouping and reporting layer does not alter the NN model, optimizer, loss,
grid, seed rule, annual windows, precision or existing training checkpoints.

`annual/combined/nn_training_summary.csv` puts each selected member's initialization,
retained epoch and last executed epoch side by side. `retained_train_mse` and
`retained_validation_mse` are evaluation-mode, unpenalized MSEs at that member's
best validation epoch, not the final epoch or the online penalized training loss.
`best_validation_loss` in annual metrics remains the **ensemble** validation MSE;
it need not equal the mean of member validation MSEs. L1 is applied to dense kernels
in the training objective; nonzero weights alone do not establish useful learning.

Every retained epoch must agree with its saved validation history and seed/member
identity. Flags include zero parameter update, all dense weights zero, constant
training predictions and retained training MSE not below initialization. These
are diagnostic flags, not automatic exclusion rules. Negative out-of-sample R2
is reported unchanged. Review forecast dispersion and zero/training-mean baseline
comparisons alongside learning curves before interpreting economic performance.

`year_sources.csv` links the copied per-year `NN1_training.csv` (all grid
candidates and members) and `NN1_training.png` (selected ensemble curves), using
paths relative to the combined experiment directory. `original_*` columns retain
the original paths. Collection copies complete source results under `sources/`
with verified file hashes and leaves every original experiment unchanged. Existing
combined runs gain these copies on their next collect; download that one complete
combined folder for analysis. The copied checkpoint is byte-identical to the
original and contains all candidate histories and selected ensemble weights.
Collection validates and pools forecasts without retraining or loading features.
Histories are still saved at annual completion; this change does not provide
live epoch streaming or within-epoch restart.

## Sources

- [Published paper](https://academic.oup.com/rfs/article/33/5/2223/5758276), Section 1.7 and Equations (17)-(18).
- Local `docs/online_appendix.pdf`, printed pages 10 and 13: B.3 / Algorithms 5-7 and Table A.5.
- [Author-linked public repository](https://github.com/xiubooth/ML_Codes): simulation codes; not treated as recovered empirical NN code.
- [Official PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/), v2.13.0 CUDA 13.0.
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- [uv and PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
- [SCRP Slurm](https://scrp.econ.cuhk.edu.hk/guide/slurm).
- [AI Hub organization guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide).

## Verification and pilot results

2026-10-24: installed PyTorch 2.13.0+cu130 and verified a real CUDA
forward/backward operation on the local RTX 5080 (sm_120 supported).
All 220 tests passed in 42.861 seconds, including five model-level NN tests,
two annual NN workflow tests and GPU-aware launcher checks. The first full-suite
attempt exposed five outdated CPU-only concurrency expectations in a shared
launcher test; those expectations now account for the single-GPU NN default.
No numerical training code was changed in response to out-of-sample results.

### NN1, fixed 5% sample, test year 1987

Completed experiment: `nn1_p05_1987-1987_20261024T131639296733Z` under
`results/experiments/`. Ran the full six-candidate grid and ten seeds per candidate
on the RTX 5080. Training: 22,395 stock-months in 1957-1974; validation: 37,680
in 1975-1986; test: 3,892 in 1987. All 920 prepared features are retained.

| Quantity | Observed result |
| --- | --- |
| Selected learning rate / L1 | 0.01 / 0.001 |
| Ensemble validation MSE | 0.0280647043313 |
| Zero-forecast validation MSE | 0.0270368751671 |
| Training-mean validation MSE | 0.0272017355087 |
| Test zero-benchmark R2 | **-2.3678928274%** |
| Training-mean forecast test R2 | +0.2165549110% |
| Test forecast mean / standard deviation | -0.0043434380 / 0.0332640907 in decimal units |
| Selected seed best epochs | 22-64; all stopped by patience |
| Selected seed parameter change L2 norms | 7.7285-7.9253 |
| Nonzero dense weights per selected member | 29,472 of 29,472 |
| Selected-seed initial training MSE range | 0.3919-0.9320 |
| Selected-seed retained-epoch training MSE range | 0.01659-0.02763 |
| Tuning/prediction time before checkpoint writing | 291.52 seconds |
| Observed peak Python resident RAM | 1.771 GiB |
| Peak PyTorch CUDA allocated memory | 144.92 MiB; excludes CUDA context/driver memory |
| Restored CPU vs saved CUDA test predictions | All 3,892 verified; max absolute difference 3.532e-7 |

The real completed experiment was also passed through `resume`; completed years
are recognized without refitting. Restoration evidence is saved as
`nn_restore_verification.json` inside the annual run directory.

This run learns and produces varying predictions, but loses to a zero forecast
on both validation and test data. It is not successful out-of-sample replication.
Both selected hyperparameters are at the configured upper endpoints; record this
without expanding the grid after looking at the test score. The small training
sample supplies only three gradient updates per epoch at batch size 10,000,
compared with about 48 for the full-universe 1987 training window. The pilot
does not establish performance of the full sample or the other architectures.

### NN3, the same fixed 5% sample and test year 1987

Completed experiment: `nn3_p05_1987-1987_20261024T132334325467Z`, using the
same six candidates and ten seeds. This architecture was chosen in advance as
the paper's main deeper specification; do not use this one test-year comparison
to select an architecture for the full experiment.

| Quantity | Observed result |
| --- | --- |
| Selected learning rate / L1 | 0.01 / 0.001 |
| Ensemble validation MSE | 0.0271088432305 |
| Test zero-benchmark R2 | **-0.6058561232%** |
| Test forecast mean / standard deviation | 0.0031802377 / 0.0143875811 in decimal units |
| Selected seed best epochs | 19-40; all stopped by patience |
| Selected seed parameter change L2 norms | 8.4041-9.0918 |
| Nonzero dense weights per selected member | 30,088 of 30,088 |
| Selected-seed initial training MSE range | 0.04548-1.46061 |
| Selected-seed retained-epoch training MSE range | 0.01591-0.02892 |
| Tuning/prediction time before checkpoint writing | 178.17 seconds |
| Observed peak Python resident RAM | 1.771 GiB |
| Peak PyTorch CUDA allocated memory | 144.94 MiB; excludes context/driver memory |
| Restored CPU vs saved CUDA test predictions | All 3,892 verified; max absolute difference 1.550e-7 |

The regularized NN3 forecast improves upon the training-mean validation forecast
but still has slightly greater validation MSE than zero. Its 1987 test R2 is also
negative. All selected seeds reduced training MSE and changed their parameters;
the previous reported all-zero training failure was not reproduced. A completed
real `resume` check skipped the saved year without refitting.

These checks establish implemented and testable training mechanics on this local
GPU, not a successful full-period replication. NN2/NN4/NN5 have configuration and
architecture tests but have not been fitted to the real panel in this work.
SCRP/AI Hub submission paths were tested with mocked Slurm; no remote jobs were
submitted. A small compute-node GPU check and full-sample experiments remain.
