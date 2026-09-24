# GKX panel preparation

Run separately before the model workflow:

```powershell
python -B -m core.data_cleaning.prepare_gkx
```

Defaults: March 1957 through December 2016; reads only `data/raw/datashare.csv`,
`macro_predictor.csv`, `risk_free_rate_fama_french.csv`, and `monthly_raw.csv`.
Writes `data/processed/gkx/panel.csv`, `metadata.json`, and
`preparation_report.csv`. The existing `load_dataset("gkx")` consumes this format.
Existing output directories are refused. A failed build may retain partial CSVs,
but has no completion metadata and cannot be loaded as a completed dataset.
Use a new output directory for a retry or inspect and remove the failed output manually.

For a small real-data trial:

```powershell
python -B -m core.data_cleaning.prepare_gkx --start-month 1957-03 --end-month 1957-04 --output-dir data/processed/gkx_trial
```

The small trial still scans the raw sources for categories/target rows. Raw files
are never modified; `--raw-dir`, `--output-dir`, and `--chunksize` are supported.
The characteristic CSV must be sorted by month. Complete months are carried
across read chunks before ranking. The wide panel is appended monthly; the
narrow CRSP target table is held in memory. The current model loader still
loads the entire completed panel and converts predictors to dense float64.

## Alignment and target

- GKX DATE identifies return_month. feature_month = return_month minus one month.
- Macro observations join on feature_month; CRSP ret and FF RF join on return_month.
- FF RF is in percent: excess_return = ret - RF/100. `retx` is not used.
- No additional delisting return adjustment is applied. This is a limitation of
  this first preparation version, not a claim about the authors' exact treatment.
- Duplicate keys, missing macro/RF, invalid nonmissing SIC, and unexpected nonnumeric returns fail.
- Missing target returns are excluded and counted; targets are never imputed.

## Explicit preprocessing choices

For each full GKX monthly cross-section, impute raw characteristics with their
cross-sectional medians, then average-rank ties and map ranks by
`2*(rank-1)/(n-1)-1`. Entirely missing columns, constant columns, and singleton
cross-sections map to zero. These are specified implementation conventions;
the paper establishes ranking and median imputation but not this complete recipe.
Do this before dropping missing targets, so return availability does not determine ranks.
Entirely missing columns and missing-cell counts appear in the monthly report.

Keep 94 ranked characteristics and their interactions with dp, ep, bm, ntis,
tbl, tms, dfy, and svar. Formulas are recorded in metadata. No extra standalone
macro columns or global standardization are added.

Retain all SIC dummies with a fixed vocabulary from the selected GKX sample.
The vocabulary uses category labels only, not returns. Feature count is
846 plus the observed number of industries; 920 is not forced.

Metadata records ordered column roles, source paths/sizes/modification times,
package versions, cleaning conventions, row count, and the output SHA-256.
Source sizes/times are checked again before completion. These source records
are not cryptographic input hashes. Preserve raw data vintages separately.

Focused tests cover target-month alignment, RF units, interactions, imputation,
ties, missing targets, duplicate keys, chunk boundaries, and the unchanged loader.
Matching published performance is corroboration, not a substitute for these checks.

## Missing SIC: agreed baseline and follow-up

Decision agreed with Phil on 2026-09-21. This is an explicit replication
assumption; GKX's exact treatment of missing industry classifications has not
been established from the authors' empirical code or documentation.

### Evidence

The initial preparation run stopped because missing SIC was treated as an
error. Inspection found missing codes, not malformed nonmissing codes.

- Full supplied file, January 1957 through December 2021: 4,117,300 rows,
  32,793 PERMNOs, 74 observed SIC values, and 300,359 missing-SIC rows.
- Of those PERMNOs, 5,693 always have missing SIC, 4,564 have both missing and
  observed SIC, and 22,536 never have missing SIC within this file.
- Paper-period selection, March 1957 through December 2016: 254,864 of
  3,760,033 rows have missing SIC (approximately 6.8%).
- Reproducible inspection script: `core/raw_data_examinine/data_check_sic.py`.
  Detailed CSV reports are under `data/sic_inspection/`. Full-file counts
  must not be confused with the paper-period counts.

### Approved treatment

1. Keep observations with missing SIC, subject to the other existing eligibility
   rules such as having a valid target return.
2. Preserve missing `sic2` as NA in the source/audit data. Never modify raw data.
3. In the model matrix, set every industry dummy to zero for missing SIC.
   Retain all 74 observed industry dummies: dropping a reference dummy would
   make missing SIC indistinguishable from that reference industry.
4. Keep a `sic_missing` audit flag outside `feature_columns`; report missing-SIC
   counts before and after target exclusions. Neither the raw SIC column nor
   the audit flag is a numeric model predictor.
5. Do not assign an artificial numeric industry such as 0 or 9999999, impute
   a median/mode SIC, forward-fill, backward-fill, or drop stocks solely for
   missing SIC. Code 75 already occurs in the data and is not a missing label.
6. The baseline retains 846 continuous features plus 74 industry dummies, for
   920 features. Missing SIC introduces no additional predictor.

### Implementation status and future investigation

Implemented in `prepare_gkx.py` on 2026-09-21. Missing SIC is excluded from
the industry vocabulary and encoded with all-zero dummies. Panel columns
`sic2` (nullable) and `sic_missing` (0/1) are audit-only and excluded from
`feature_columns`. Monthly reports include `missing_sic_input_rows` and
`missing_sic_output_rows`. Metadata records this treatment. Invalid nonmissing
codes still fail validation.

Verification: all 76 tests passed. A real March-April 1957 build saved 2,117
rows, including all 1,125 rows with missing SIC, and passed the unchanged
dataset loader. Missing-SIC rows had all-zero industry dummies; known-SIC
rows had exactly one active industry dummy. The trial's 53 observed industries
give 899 features; the full paper-period vocabulary has 74 industries (920
features). Trial outputs are in `data/processed/gkx_sic_validation/`.
This verification did not build the full 1957-2016 wide panel.

A later sensitivity check may add a separate `sic2_missing` predictor (921
features for the same vocabulary). That is not the approved baseline.

Investigate historical CRSP SIC coverage separately. Preserve the original
GKX codes and record any recovered code, source, and effective dates separately.
Do not automatically substitute newer CRSP classifications into the baseline.
If recovery is pursued, compare with observed GKX codes and establish a
forecast-date-safe matching convention; never apply the latest classification
to the entire history. Missing-code recovery remains future work.
