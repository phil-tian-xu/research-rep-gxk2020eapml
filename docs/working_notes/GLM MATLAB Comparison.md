# GLM Python–MATLAB comparison

This is a small numerical reference check before implementing the empirical GLM
workflow. It uses artificial data, not the 5% panel or the full GKX panel. It does
not change `main.py`, model registration, experiment configs, or cluster jobs.
The earlier `core/models/group_lasso.py` draft is not used by this comparison.

## Run it

From the repository root in PowerShell, using the project Python environment:

```powershell
python -B -m core.data_examine.glm_matlab prepare --output results/glm_matlab/reference_v1
```

This runs the Python reference and creates one directory with shared inputs,
Python outputs, source snapshots, and a manifest. It refuses to overwrite an
existing directory. For a new attempt use `reference_v2`, or omit `--output` for
a timestamped directory. Preparation takes seconds and needs no MATLAB license.

Download the [authors' repository](https://github.com/xiubooth/ML_Codes) using
GitHub's **Code → Download ZIP**, or use your existing checkout. Keep the original
`Simu_Matlab` functions unchanged. No need to execute `DGP.m` or `MC_Reg.m`.

Once MATLAB is installed, run this in its Command Window. Change `authorDir` to
the actual extracted directory on your machine:

```matlab
cd('E:/Workstation/coding_projects/coding_research/paper_replication/research-rep-gxk2020eapml');
runDir = fullfile(pwd, 'results', 'glm_matlab', 'reference_v1');
authorDir = 'C:/Users/phil_prod/Downloads/ML_Codes-master/Simu_Matlab';
addpath(runDir);
run_glm_reference(runDir, authorDir);
```

The supplied driver invokes the original authors' functions, not a second port
of the Python optimizer. It records their source text and the MATLAB version in
`matlab_results.mat`. Required functions are `cut_knots_degree2`, `proximal`,
`proximalH`, `f_grad`, `f_gradh`, `soft_threshodg`, `lossh`, and `fw1`.
Missing or shadowed functions produce a clear error. It refuses to overwrite
an existing MATLAB output. The full authors' simulation has extra dependencies
that this bounded driver does not use.

Then return to PowerShell:

```powershell
python -B -m core.data_examine.glm_matlab compare --run-dir results/glm_matlab/reference_v1
```

Read `comparison.csv` for individual checks and `comparison.json` for the overall
PASS/FAIL result and provenance. A numerical mismatch exits with code 1. Missing
MATLAB output raises an error; it is never treated as a passed comparison.
Share those two reports if anything fails.

## What is compared

The fixture has 240 training, 100 validation, and 120 test observations; four
inputs expand to 20 columns. It includes nonlinear signal, return outliers,
tied predictor values, and a constant input. Python generates the data once,
and MATLAB loads the same MAT file, avoiding cross-language RNG differences.

Checks include knots, both standardization scales, centered and standardized
basis matrices, the group proximal operator, all candidate coefficients and
validation scores, selected penalty, squared and Huber coefficients, group
norms, objectives, stationarity residuals, predictions, and R². MATLAB must echo
exactly the same fixture. Missing fields, shape mismatches, and nonfinite values
fail. Selection is exact; transformations use atol/rtol 1e-10; iterative outputs
use atol 1e-7 and rtol 1e-5. Tolerances are recorded per field, not widened on failure.

## Reference conventions — not an approved empirical specification

The comparison follows the GLM section of
[MC_Reg.m](https://github.com/xiubooth/ML_Codes/blob/master/Simu_Matlab/MC_Reg.m):
training standard deviations (sample denominator n−1), training knots at the
minimum and quartiles, another training-based scaling after expansion, and fixed
training-target-mean intercept. The penalty grid is `10^(.5:.1:3)`; squared-loss
validation selects the penalty, then Huber refines that fit at the same penalty.
The simulation threshold is 0.2 and iterate tolerance is 1e-10.

[cut_knots_degree2.m](https://github.com/xiubooth/ML_Codes/blob/master/Simu_Matlab/cut_knots_degree2.m)
creates `[x, (x−t1)^2, (x−t2)_+^2, (x−t3)_+^2, (x−t4)_+^2]` per input and
centers each supplied dataset separately. We reproduce that behavior here,
including evaluation-set centering. This is deliberately not the proposed
training-only centering adaptation. All five columns form one unweighted group.

MATLAB's default quantile interpolation uses probabilities `(i−.5)/n`; Python
uses `np.quantile(..., method="hazen")`, not NumPy's default. See
[MathWorks quantile documentation](https://www.mathworks.com/help/matlab/ref/quantile.html).

The optimization objective is a **sum** of half-squared/half-Huber losses plus
lambda times the sum of group norms. There is no division by sample size, no
factor-of-two change to the gradient, and no group-size weight. The reference
returns the accelerated iterate and uses the original 30,000/3,000 iteration
limits. Iteration counts and whether the iterate-change stop was met are in
`manifest.json`; small changes between iterates do not prove stationarity.
Objective and stationarity checks make this limitation visible. See
[proximal.m](https://github.com/xiubooth/ML_Codes/blob/master/Simu_Matlab/proximal.m)
and [proximalH.m](https://github.com/xiubooth/ML_Codes/blob/master/Simu_Matlab/proximalH.m).

Reported reference R² uses the training-mean benchmark, as in the simulation.
It is **not** the zero-benchmark Table 1 statistic used by our empirical runs.
Industry-dummy treatment, empirical threshold construction, and empirical
intercept/centering/tuning decisions remain separate from this numerical check.

## Verification status

Python tests cover hand-calculated splines and block shrinkage, MATLAB quantile
examples, an independent SciPy constrained Huber/group-lasso solution, iteration
limit reporting, comparison failures, and the full small Python path. MATLAB
execution has now been completed by Phil using R2026b. The saved `reference_v1`
report passes all checks; the maximum Huber test-prediction difference is
4.44e-16. This verifies the reference fixture, not the empirical adaptations.

```powershell
python -B -m unittest discover -s tests -p test_glm_matlab.py -v
```
