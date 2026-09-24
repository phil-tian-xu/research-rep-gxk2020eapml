I would keep your current top-level structure and build a straightforward research pipeline inside it. The main addition should be a clear progression from source data to a monthly modeling panel, model estimation, saved predictions, and evaluation.
Everything below is a proposal. I have not created or changed anything.
1. What currently exists
I reviewed the Python scripts, project settings, documentation structure, and data filenames. I did not load the large datasets or inspect materials/.
Existing component	Current role
[data_prepare.py](E:/Workstation/coding_projects/coding_research/paper_replication/research-rep-gxk2020eapml/core/data_prepare.py)	Runs CRSP extraction in yearly batches and saves raw/intermediate outputs with metadata
[wrds_data_query.py](E:/Workstation/coding_projects/coding_research/paper_replication/research-rep-gxk2020eapml/core/data_query/wrds_data_query.py)	Queries CRSP returns, delistings, and trading proxies; constructs inspection fields
[data_check.py](E:/Workstation/coding_projects/coding_research/paper_replication/research-rep-gxk2020eapml/core/data_cleaning/data_check.py)	Splits large source CSVs into monthly files
[data_check_compare.py](E:/Workstation/coding_projects/coding_research/paper_replication/research-rep-gxk2020eapml/core/data_cleaning/data_check_compare.py)	Harmonizes Green/Xiu column names and selects the characteristic columns
Coverage scripts	Compare stock membership and time coverage
Green SAS scripts	Original/reference characteristic construction and your adapted version
Documentation	Paper, appendix, field dictionary, macroeconomic construction notes, and technical notes
[main.py](E:/Workstation/coding_projects/coding_research/paper_replication/research-rep-gxk2020eapml/main.py)	Currently an empty entry point


Your data folders already distinguish raw and intermediate CRSP data. They also contain monthly CSV, reshaped CSV, and Parquet directories.
The main missing components are the final modeling panel, model implementations, annual estimation loop, and evaluation pipeline.
2. Proposed folder and file layout
The following tree is rooted at:
E:\Workstation\coding_projects\coding_research\paper_replication\research-rep-gxk2020eapml
Existing files are retained. New files should be added as their functionality is implemented, rather than creating every empty file now.
research-rep-gxk2020eapml/
├── main.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── AGENTS.md
├── PROJECT_AGENTS.md
├── README.md
│
├── config/
│   ├── gkx_monthly.toml
│   └── smoke.toml
│
├── core/
│   ├── data_prepare.py                 # Existing CRSP extraction entry point
│   │
│   ├── data_query/
│   │   └── wrds_data_query.py          # Existing
│   │
│   ├── raw_data_query_Green2017/
│   │   ├── GHZ_20161231_DataSetUp.sas   # Existing reference
│   │   └── green_raw_data_query_v1.sas # Existing adaptation
│   │
│   ├── data_cleaning/
│   │   ├── data_check.py               # Existing
│   │   ├── data_check_compare.py       # Existing
│   │   ├── data_check_stock_coverage.py
│   │   ├── data_check_time_coverage.py
│   │   └── build_panel.py              # Assemble eligible monthly observations
│   │
│   ├── features.py                    # Characteristics, macro inputs, interactions
│   ├── preprocessing.py               # Ranking, imputation, fitted transforms
│   ├── splits.py                      # Annual training/validation/test boundaries
│   ├── training.py                    # Tuning and annual estimation
│   │
│   ├── models/
│   │   ├── linear.py                  # OLS, robust linear, lasso/ridge/ENet
│   │   ├── dimension_reduction.py     # PCR and PLS
│   │   ├── spline.py                  # Spline GLM with group lasso
│   │   ├── trees.py                   # RF and GBRT
│   │   ├── neural.py                  # NN1–NN5 and NN fitting
│   │   ├── losses.py                  # Shared loss/penalty definitions
│   │   └── optimizers.py              # Shared proximal optimization
│   │
│   └── evaluation/
│       ├── prediction.py              # OOS R² and forecast comparisons
│       ├── portfolios.py              # Portfolio formation and performance
│       ├── importance.py              # Variable importance and response curves
│       └── reporting.py               # Paper-style tables and figures
│
├── utils/
│   ├── path_info.py                   # Existing; keep as the path utility
│   └── io.py                          # Shared saving/loading, when needed
│
├── data/
│   ├── raw/
│   ├── intermediate/
│   └── processed/
│
├── results/
│   ├── diagnostics/
│   └── runs/
│
├── docs/
│   ├── main_paper.pdf
│   ├── online_appendix.pdf
│   └── working_notes/
│       ├── ...existing notes...
│       └── Replication Decisions.md
│
├── tests/
│   ├── test_panel.py
│   ├── test_preprocessing.py
│   ├── test_splits.py
│   ├── test_models.py
│   └── test_evaluation.py
│
└── materials/                         # Existing exploratory workspace
The current monthly CSV/Parquet folders can stay where they are while the pipeline develops. We can consolidate their role later without disrupting existing scripts.
3. Define the monthly panel before implementing models
The most important design decision is what a modeling observation means.
I propose that each row represents:
One security, information available at a particular month-end, and its subsequent realized return.

A useful panel would contain:
Field	Purpose
permno	Security identifier
feature_month	Month when predictors are available
return_month	Month of the target return
excess_return	Realized target, in decimal units
market_cap	Formation-date size for portfolio weights and size subsets
sic2	Industry identifier
94 characteristic columns	Predictors with documented timing
Eligibility/availability flags	Explain inclusion, exclusion, and missing observations


The eight macroeconomic series can remain in a separate monthly table and be joined when constructing model inputs.
build_panel.py would handle source alignment, return construction, eligibility, and merges. features.py would produce the ordered feature matrix, including characteristic–macro interactions and industry indicators.
Two safeguards follow directly from your current code:
- Padded CRSP rows are not automatically eligible observations. Your extraction code deliberately creates missing rows outside observed security histories.
- Already-lagged characteristics must not be lagged again automatically. Green-derived data and author-provided data need explicit source conventions.
This is also where we keep characteristics used for prediction separate from raw market capitalization used for portfolio weights.
4. Separate monthly transformations from transformations fitted on training data
I would put preprocessing in one module, but make the distinction explicit:
Transformation type	Examples	When computed
Available-month cross-sectional operations	Characteristic ranks and monthly median imputation	Using that month’s eligible predictor observations
Fitted transformations	Any chosen standardization, spline knots, component extraction	Estimated using the permitted training sample
Model-input construction	Macro interactions and indicator columns	Applied consistently to each split


This avoids two common errors:
1. Fitting a scaler or PCA on the entire sample before splitting.
2. Treating same-month cross-sectional ranking as though it must use historical training observations only.
The feature list and column order should have one authoritative definition, initially in features.py. Your existing 94-characteristic list provides the starting point.
For storage, I would retain the compact characteristic panel and construct the 920-feature matrices as needed. Storing multiple complete copies of all interactions for every model and year would consume substantial memory and disk space.
3. Keep model code organized by method family
I would use one NN implementation whose widths are supplied as a setting:
hidden_sizes = [32, 16, 8]  # NN3
That same implementation supports NN1–NN5. There is no need for five separate NN files.
Likewise:
- linear.py shares the linear prediction logic across loss and penalty choices.
- dimension_reduction.py contains PCR and PLS.
- trees.py contains forest and boosting implementations.
- losses.py centralizes the paper’s loss conventions.
- optimizers.py contains proximal optimization shared by ENet and the spline model.
Model functions should accept explicit arrays and settings. They should not query WRDS, infer sample dates, or decide where to write results.
A small amount of common calling convention is useful:
fitted_model = fit_model(X_train, y_train, settings)
predictions = fitted_model.predict(X_test)
I would introduce only the small adapters needed to make that work. We do not need a general-purpose modeling framework or an elaborate class hierarchy.
6. Give the annual training loop one owner
training.py should coordinate:
Select test year
    ↓
Get training and validation windows
    ↓
Prepare inputs using the correct information sets
    ↓
Fit candidate configurations
    ↓
Evaluate validation predictions
    ↓
Select configuration and retain the fitted model
    ↓
Predict the test year
    ↓
Save results and mark that model-year complete
splits.py would contain the date logic independently, making it easy to test.
Model-specific optimization stays with each model. For example, NN minibatches, early stopping, and checkpoint restoration belong in neural.py; the annual loop should not reproduce that logic.
The natural resumable unit is one model and one test year. A failed NN fit for 1998 should not require rerunning completed years or the other models.
A run should resume only when its dataset version and settings match the saved metadata.
7. Use small configuration files
I recommend two initial configurations:
Configuration	Purpose
gkx_monthly.toml	Intended monthly replication settings
smoke.toml	A deliberately small run to verify the pipeline


The full configuration would record:
- Dataset source/version.
- Sample and split dates.
- Requested models.
- Exact hyperparameter grids.
- Loss and preprocessing choices.
- NN seeds and ensemble size.
- Device and CPU parallelism.
- Output/run identifier.
The smoke configuration would reduce the sample, candidate grids, and number of ensemble members. Its results must be clearly identified as pipeline checks, not replication estimates.
Unresolved research choices should not be silently filled by software defaults. The configuration records the selected value; Replication Decisions.md explains whether it comes from the paper, author code, or our own decision.
8. Organize outputs by experiment
Within the proposed results directory:
results/runs/<run_id>/
├── config.toml
├── metadata.json
├── run.log
├── validation/
│   └── nn3_1987.parquet
├── predictions/
│   └── nn3_1987.parquet
├── checkpoints/
│   └── nn3_1987/
├── metrics/
├── tables/
└── figures/
The most valuable saved outputs are:
Output	What to retain
Run metadata	Dataset identity, code revision, environment information, seeds, device, completion status
Validation results	Every candidate’s settings and score; NN stopping information
Selected configuration	Chosen hyperparameters for each model-year
Test predictions	Security ID, feature month, return month, model, prediction, actual target
Checkpoints	Selected models and fitted preprocessing; all selected NN ensemble members
Reports	Metrics and figures derived from saved predictions


Evaluation should operate on saved predictions whenever possible. Changing a portfolio report or correcting a metric should not trigger model retraining.
For long NN runs, saving the best candidate checkpoints during tuning avoids having to refit the winning candidate with a different random trajectory.
9. Preserve the current data work
Your current files suggest two useful research paths:
- Author-provided characteristics: use them to establish a modeling benchmark.
- Green-based reconstruction: use your SAS and CRSP work to investigate data-construction differences.
I recommend keeping both paths and making characteristic_source explicit in the configuration.
A useful first milestone would be a modeling benchmark using the author-provided file, after verifying its columns, timing, and coverage. That helps distinguish model-implementation differences from characteristic-reconstruction differences. It does not replace the eventual data replication.
I would also keep raw extracts unchanged. New harmonized and modeling datasets should go into intermediate and processed storage with identifiable versions.
10. Tests and implementation order
Tests should focus on research errors that can quietly contaminate results:
Area	Important checks
Panel	Unique security-month keys, correct return alignment, exclusion of synthetic observations
Preprocessing	Known rank/imputation examples and training-only fitted transforms
Splits	Exact initial windows, annual movement, no overlapping training/validation/test observations
Models	Huber and proximal calculations; selected checkpoint restoration
Evaluation	Zero-benchmark $R^2$, portfolio weights, hand-calculated return examples


I would implement the project in this order:
1. Define the panel and document source timing.
2. Build panel preparation and split tests.
3. Run a small OLS-3 benchmark through training and saved-prediction evaluation.
4. Add PCR, PLS, and RF using the same pipeline.
5. Add the custom Huber/penalized methods and spline GLM.
6. Add NN1, validate its training behavior, then enable NN2–NN5 and ensembles.
7. Expand to the full monthly experiment, portfolio analysis, and variable importance.
One reproducibility issue I noticed: your current [.gitignore](E:/Workstation/coding_projects/coding_research/paper_replication/research-rep-gxk2020eapml/.gitignore) excludes uv.lock. I would propose tracking the lockfile while continuing to exclude datasets, model checkpoints, and generated results.
The first concrete design discussion should therefore be the modeling-panel definition and which characteristic source supplies the initial benchmark. Those decisions determine the interfaces for nearly everything that follows.