GKX selects the eight predictors in **Section 2.1, p. 2248**. Their definitions come from **Welch and Goyal (2008), Section 1, pp. 1457–1459**.

Use the **Monthly** worksheet of `PredictorData2025.xlsx`. Below, \(t\) denotes the observation month and \(\ln\) denotes the natural logarithm.
## 1. Dividend–price ratio (`dp`)
> “The Dividend Price Ratio (d/p) is the difference between the log of dividends and the log of prices.”
$$dp_t=\ln(\texttt{D12}_t)-\ln(\texttt{Index}_t)$$

`D12` already contains the trailing 12-month dividend sum. Use the current month’s index level. **Source: p. 1457.**

## 2. Earnings–price ratio (`ep`)
> “The Earnings Price Ratio (e/p) is the difference between the log of earnings and the log of prices.”
$$ep_t=\ln(\texttt{E12}_t)-\ln(\texttt{Index}_t) $$

`E12` already contains the trailing 12-month earnings sum. **Source: pp. 1457–1458.**
## 3. Book-to-market ratio (`bm`)
> “The Book-to-Market Ratio (b/m) is the ratio of book value to market value for the Dow Jones Industrial Average.”
$$ bm_t=\texttt{b/m}_t $$

Use the supplied ratio directly. Its construction uses previous-year-end book value for March–December and book value from two year-ends earlier for January–February. Do not apply those lags again. **Source: p. 1458.**
## 4. Net equity expansion (`ntis`)
> “Net Equity Expansion (ntis) is the ratio of 12-month moving sums of net issues by NYSE listed stocks divided by the total end-of-year market capitalization of NYSE stocks.”

The underlying net-issuance calculation is:

$$ \operatorname{NetIssue}_t = \operatorname{Mcap}_t- \operatorname{Mcap}_{t-1}(1+\operatorname{vwretx}_t), $$
where `Mcap` is NYSE aggregate market capitalization and `vwretx` is its value-weighted return excluding dividends.

The workbook already provides the constructed predictor:

$$ ntis_t=\texttt{ntis}_t $$
**Source: p. 1458, Equation (3).**
## 5. Treasury-bill rate (`tbl`)
> “Treasury-bill rates from 1934 to 2005 are the 3-Month Treasury Bill: Secondary Market Rate from the economic research data base at the Federal Reserve Bank at St. Louis (FRED).”
$$ tbl_t=\texttt{tbl}_t $$
Retain the supplied yield units. `Rfree` is a separate series, not a substitute for this predictor. **Source: p. 1459.**

## 6. Term spread (`tms`)
> “The Term Spread (tms) is the difference between the long term yield on government bonds and the Treasury-bill.”
$$ tms_t=\texttt{lty}_t-\texttt{tbl}_t $$

Use the yield `lty`, not the return `ltr`. **Source: p. 1459.**

## 7. Default yield spread (`dfy`)
> “The Default Yield Spread (dfy) is the difference between BAA and AAA-rated corporate bond yields.”
$$ dfy_t=\texttt{BAA}_t-\texttt{AAA}_t $$
The subtraction order is BAA minus AAA. **Source: p. 1459.**

## 8. Stock variance (`svar`)
> “Stock Variance is computed as sum of squared daily returns on the S&P 500.”

At monthly frequency:
$$ svar_t=\sum_{d\in\text{month }t}r_{\mathrm{S\&P500},d}^{\,2}. $$
Use the already constructed workbook column:
$$ svar_t=\texttt{svar}_t $$
Do not take its square root or replace it with the squared monthly return. **Source: p. 1458.**

## Implementation summary

| Predictor | Workbook expression   |
| --------- | --------------------- |
| `dp`      | `ln(D12) - ln(Index)` |
| `ep`      | `ln(E12) - ln(Index)` |
| `bm`      | `b/m`                 |
| `ntis`    | `ntis`                |
| `tbl`     | `tbl`                 |
| `tms`     | `lty - tbl`           |
| `dfy`     | `BAA - AAA`           |
| `svar`    | `svar`                |

- Parse `yyyymm` as the monthly merge key; check duplicates and missing months.
- Check positivity before logarithms and consistent yield units before subtraction.
- Keep aggregate `bm` and `ep` distinct from firm-level characteristics.
- Match month-\(t\) predictors to month-\(t+1\) target returns.
- Preserve the source workbook and record its vintage. Historical values in the 2025 update may differ from those used by GKX.

## References
Gu, S., B. Kelly, and D. Xiu. 2020. “Empirical Asset Pricing via Machine Learning.” _Review of Financial Studies_ 33(5): 2223–2273.
Welch, I., and A. Goyal. 2008. “A Comprehensive Look at the Empirical Performance of Equity Premium Prediction. _Review of Financial Studies_ 21(4): 1455–1508.