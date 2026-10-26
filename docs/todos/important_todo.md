# Important todos

## Re-run experiments on Green-based characteristics

Added: 2026-10-26. Status: pending.

- [ ] Construct firm characteristics using Green's approach and re-run the
  experiments, starting with OLS-3 + Huber and then the broader models.

Two motivations:

1. Check whether GKX-specific construction or preprocessing choices make
   predictive performance look better, including any undocumented choices or
   potential look-ahead bias. Investigate why values missing in Green-based data
   are available in Xiu's dataset before imputation.
2. Establish a reproducible Green-based pipeline that we can extend to the latest
   available data for future experiments.

Compare common stock-months using the same returns, preprocessing, model settings,
and evaluation periods to isolate characteristic-source effects. The contribution
of data differences to the current replication gaps remains unresolved; similar
OLS-3 performance alone does not rule it out.
