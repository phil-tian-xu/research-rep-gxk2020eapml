GKX replication: data alignment and feature construction
- GKX DATE labels the target return month. The 94 firm characteristics already incorporate lag adjustments. Do not lag them again.
- Goyal–Welch yyyymm labels the macro observation month. Pair the previous month’s macro observations with each GKX row.
- Define return_month = month(GKX.DATE) and feature_month = return_month - 1 calendar month.
Component	Example using the first GKX row
Security	permno = 10006
GKX row	DATE = 19570131
Forecast formation month	December 1956
Macro row to use	yyyymm = 195612
CRSP target return	January 1957
Target $y$	January stock total return minus January risk-free return


Construct eight macro predictors from the selected macro row:
Predictor	Construction
dp	ln(D12) - ln(Index)
ep	ln(E12) - ln(Index)
bm	b/m
ntis	ntis
tbl	tbl
tms	lty - tbl
dfy	BAA - AAA
svar	svar


Construct model inputs:
- Cross-sectionally rank firm characteristics each month and map to $[-1,1]$; handle missing characteristics using the paper’s monthly cross-sectional median rule.
- Include 94 characteristics, their $94\times8$ macro interactions, and SIC industry dummies: 920 predictors in the paper’s sample.
- Treat sic2 as categorical. Do not include permno, dates, or the target as predictors.
- Keep macro bm and ep distinct from firm-level characteristics with the same names.
Implementation checks:
- Join by calendar month, not exact month-end day or row position. Merge CRSP by permno + return_month and macro data by feature_month.
- Use consistent decimal units for stock and risk-free returns. Specify delisting-return treatment explicitly.
- Check duplicate keys, unmatched months, missing macro values, and nonpositive inputs to logarithms; do not silently fill missing macro observations.
- The supplied January 1957 row illustrates alignment; the paper’s empirical sample starts in March 1957.
- Evidence: GKX Equation (21), its README’s same-month return matching instruction, the local mom1m–CRSP spot check, and the user’s macro index-date check. Other macro series’ publication timing has not been individually audited. Matching published results is corroboration, not proof of correctness.