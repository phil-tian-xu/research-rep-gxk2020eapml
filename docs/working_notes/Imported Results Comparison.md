# Imported results comparison

Run from the repository root:

```powershell
.venv/Scripts/python.exe -B -m core.compare_replication
```

Reads the four full 1987–2016 OLS-3+H, ENet+H, PCR and PLS imports under
`results/scrp_imports`. Requires exactly one experiment per model; duplicate
imports fail rather than silently selecting a run. An alternative import folder
can be supplied with `--source-root`.

Every invocation creates `results/replication_comparison/<timestamp>/`, with
local time and its UTC offset in the folder name and generation metadata:

- `table1/table1_all.csv`, `.png`, `.svg`: All-stock pooled R2, paper reference
  values from the existing reporting module, and differences in percentage points.
- `table1/annual_metrics.csv`: annual R2, observation counts, SSE and benchmark SSE.
- `figure3/enet.png`, `pcr.png`, `pls.png`, plus SVG versions and each plot's
  `*_complexity.csv`: three separate selected-complexity paths.
- `sources.json` and `README.txt`: source locations, generation time and scope.

The script reads one model-year prediction CSV at a time, verifies matching
dataset identities, observations and returns across models, requires all 12
months each year, and recomputes pooled R2 to check against the combined reports.
It does not audit every estimation assumption or checkpoint hash. Complexity
comes from the saved combined reports. No estimation or source-data loading is
needed. Top/bottom size subsamples and an HTML report are outside this workflow.
