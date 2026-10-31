# NN fitting diagnostics

`run_experiment.py diagnose-nn` reads existing checksummed NN1–NN10 fits and writes a separate HTML report, PNG figures and CSV tables. It does not train, tune, load the raw/processed feature panel, request a GPU, or change the original experiment. It requires no PyTorch import. The fitting, annual orchestration and resume fingerprints are unchanged by this reporting addition.

## Run one saved example

Main PC, PowerShell, from the project directory. This selects the original 920-input full-universe NN1 experiment, test year 2016, already imported under `results/scrp_imports`:

```powershell
.venv\Scripts\python.exe -B run_experiment.py diagnose-nn `
  nn1_full_1987-2016_20261025T072907591708Z `
  --root results/scrp_imports `
  --years 2016
```

Open the printed output directory's `index.html`, then its year link. Output defaults to a new timestamped directory under `results/nn_diagnostics`; `--output-root PATH` changes its parent. Existing reports and source experiment folders are never overwritten. Writing a report beneath its source experiment or embedded source folders is refused.

The experiment argument also accepts an absolute experiment folder, an annual experiment directory, a combined experiment ID/folder, or a fitted run containing `manifest.json`. `--root` controls ID lookup; `--aihub` selects the existing AI Hub results root. Relocated combined bundles resolve their embedded sources and check used files against the frozen SHA-256 inventory. Original feature datasets and original cluster paths are unnecessary.

`--years YEAR` or `--years FIRST:LAST` selects completed test years. Omitting it requests the experiment's full recorded horizon; incomplete or ambiguous annual fits are rejected, not silently omitted. Use an explicit completed year while other tasks are still running. The tool never submits or resumes unfinished fits. An interrupted/failed analysis retains `status=incomplete`; `index.html` at the report root and `status=complete` are published only after every requested year succeeds.

The default plots concern the validation-selected hyperparameter candidate. To inspect a different candidate, rerun the same command with `--candidate N`, using the one-based candidate number in `candidates.csv`. All candidate/seed histories and scalar summaries are exported on every invocation. Full parameter tables always concern the selected ensemble because unselected parameter tensors were not retained.

## Outputs and interpretation

| Output | Meaning |
| --- | --- |
| `summary.csv` | Reported year, dataset/input count, selected/plotted candidates, and retained-member review flag counts. |
| `methodology.json` | Completion status, selected scope, explicit thresholds, report source hash and hashes of every input manifest/config/checkpoint used. |
| `YEAR/learning_curves.png` | Evaluation-mode training/validation MSE, retained epoch dots and saved validation baselines. Curves begin at epoch 1; initialization ranges are annotated and individual initial values remain in the member table so large initial errors do not compress the trained-epoch curves. |
| `YEAR/training_objective.png` | Online half-MSE, online L1 contribution and their sum. These are batch-size-weighted averages while parameters and training-mode BN statistics change, not the objective evaluated once on the final epoch state. |
| `YEAR/optimization.png` | First-batch aggregate gradient norm for each epoch and parameter-update L2 norms from initialization to each retained state. |
| `YEAR/prediction_variation.png` | Each seed's training/validation prediction standard deviation through training, with retained epochs marked. |
| `YEAR/hyperparameters.png` | Ensemble validation MSE across learning-rate/L1 pairs, the recorded selection, candidate differences and zero/training-mean validation baselines. |
| `YEAR/parameters.png` | Exact-zero and near-zero dense-kernel fractions by layer and seed for the retained ensemble. |
| `YEAR/epochs.csv` | Every candidate/seed's recorded epoch values and the derived online objective. |
| `YEAR/members.csv` | Every candidate/seed's initialization, retained and last-executed epoch information, stop reason, update norm, nonzero counts and diagnostic warnings. |
| `YEAR/candidates.csv` | Candidate identities, ensemble validation MSE and prediction variation, and selection flags. |
| `YEAR/parameters.csv` | Selected members' per-tensor counts, exact/near-zero fractions, L1/L2 norms, RMS and absolute magnitudes. Dense kernels, biases, BN affine parameters, running buffers and counters are separate kinds; buffers/counters are not trainable parameters. |
| `YEAR/prediction_variation.csv` | Saved test-ensemble forecast mean/std/range overall and within each month, including constant/near-constant flags. It does not calculate test accuracy or compare model performance. |

CSV floating-point values use 17 significant digits; HTML parameter values use scientific notation when needed. Weight values are inspected in their recorded float32 representation. Near-zero thresholds are absolute magnitudes `1e-8`, `1e-6`, and `1e-4`, inclusive. They describe the parameters; they do not prove that a network is functionally constant, particularly with BN rescaling.

Prediction standard deviations at or below `1e-8` in decimal-return units receive a review flag. Within-month checks distinguish missing cross-sectional variation from variation over time. A month with fewer than two forecasts is marked unassessable. A zero dense kernel or an entirely zero BN scale vector is flagged separately because either removes input dependence in this sequential architecture. Expected zero biases alone are not collapse flags.

The saved model is each member's best validation epoch, not the last executed epoch. Ensemble validation MSE is not generally the average of member MSEs. The report preserves the original first-tie selection rule, selected hyperparameters, seeds and fits. Flags never remove members, alter weights, select another candidate, or trigger a refit.

## Limits

Only first-batch gradient norms were saved, including both data-loss and penalty gradients. They cannot diagnose individual layers or separate the two gradient contributions. The aggregate parameter-update norm does not supply per-layer update trajectories. Full weight/activation/gradient histories cannot be reconstructed. A maximum-epoch stop is reported as such and is not a numerical-convergence claim.

This report checks checkpoint integrity, saved identities/histories, tensor layouts and agreement between selected weight counts/norms and recorded summaries. It does not reproduce forward passes on the feature panel or establish predictive quality. It contains no cross-model test-performance comparison, portfolio analysis, or significance tests.

## Verification and worked example (2026-10-31)

All 273 tests passed, including 11 new diagnostics tests. They cover loss decomposition and ensemble/member distinctions, retained versus last epochs, exact/near-zero weights and BN buffers, zero dense/output/BN-scale cases, constant cross-sectional forecasts masked by time variation, NN6–NN10 layouts, unchanged source files/RNG/fitting identity, invalid checkpoint rejection and relocated frozen bundles.

The worked example above analyzes only 2016 of the original full-universe NN1 run, with 920 inputs. Six candidates and 60 member histories were exported. All ten retained members updated their parameters; all 29,472 dense weights per member were nonzero. None triggered the configured review flags, and no test month had near-constant cross-sectional ensemble predictions. This establishes the absence of those detected collapse symptoms for this saved fit; it does not establish predictive quality or training health for other models/years. The original experiment's 823 files retained their inventory, sizes and modification times, and the used input hashes were unchanged. Machine-readable evidence is under `results/nn_diagnostics_verification/`.
