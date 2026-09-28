"""Export All-stock Table 1 and three Figure 3 plots from imported experiments."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from core.replication_report import PAPER_ALL_R2_PERCENT


MODELS = {"ols3": "OLS3_HUBER", "enet": "ENET_HUBER", "pcr": "PCR", "pls": "PLS"}
LABELS = {"ols3": "OLS-3+H", "enet": "ENet+H", "pcr": "PCR", "pls": "PLS"}
YEARS = list(range(1987, 2017))


def make_comparison(source_root, output_root):
    sources, complexity = {}, {}
    for name, model in MODELS.items():
        matches = sorted(Path(source_root).glob(f"{name}_full_1987-2016_*/annual"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one full experiment for {name}, found {matches}.")
        sources[name] = matches[0]
        frame = pd.read_csv(matches[0] / "combined/complexity.csv").sort_values("test_year")
        if (frame.test_year.tolist() != YEARS or not frame.model.eq(model).all()
                or not frame.status.eq("complete").all()):
            raise ValueError(f"Incomplete/duplicate complexity records for {name}.")
        column = "nonzero_slopes" if name in {"ols3", "enet"} else "fitted_components"
        values = frame[column].to_numpy()
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Invalid complexity counts for {name}.")
        complexity[name] = frame

    annual, dataset_identity = [], None
    for year in YEARS:
        reference = None
        for name, model in MODELS.items():
            paths = sorted((sources[name] / "years" / str(year)).glob("*/predictions.csv"))
            if len(paths) != 1:
                raise ValueError(f"Expected one prediction file for {name}, {year}.")
            manifest = json.loads((paths[0].parent / "manifest.json").read_text(encoding="utf-8"))
            if dataset_identity is not None and manifest["dataset"] != dataset_identity:
                raise ValueError(f"Dataset identity mismatch: {name}, {year}.")
            dataset_identity = manifest["dataset"]
            panel = pd.read_csv(paths[0]).sort_values("row_position").reset_index(drop=True)
            if (panel.empty or panel.isna().any().any() or panel.row_position.duplicated().any()
                    or not panel.specification.eq(model).all() or not panel.test_year.eq(year).all()
                    or not np.isfinite(panel[["observed", "prediction"]]).all().all()):
                raise ValueError(f"Invalid predictions: {name}, {year}.")
            months = pd.PeriodIndex(panel.return_month, freq="M")
            if set(months) != set(pd.period_range(f"{year}-01", f"{year}-12", freq="M")):
                raise ValueError(f"Missing/unexpected test months: {name}, {year}.")
            identity = panel[["row_position", "return_month", "observed"]]
            if reference is not None and not identity.equals(reference):
                raise ValueError(f"Test observations differ: {name}, {year}.")
            reference = identity
            sse = float(np.square(panel.observed - panel.prediction).sum())
            denominator = float(np.square(panel.observed).sum())
            if denominator <= 0:
                raise ValueError(f"Zero benchmark denominator: {name}, {year}.")
            annual.append(dict(model=LABELS[name], test_year=year, n_test=len(panel),
                               sse=sse, benchmark_sse=denominator,
                               r2_percent=100 * (1 - sse / denominator)))
        print(f"Verified common test observations: {year}", flush=True)

    annual = pd.DataFrame(annual)
    rows = []
    for name, model in MODELS.items():
        group = annual[annual.model.eq(LABELS[name])]
        r2 = 100 * (1 - group.sse.sum() / group.benchmark_sse.sum())
        saved = pd.read_csv(sources[name] / "combined/table1_all.csv")
        if (len(saved) != 1 or saved.iloc[0].model != model
                or not np.isclose(r2, saved.iloc[0].r2_percent, rtol=0, atol=1e-10)
                or int(group.n_test.sum()) != saved.iloc[0].n_test):
            raise ValueError(f"Recomputed Table 1 disagrees with saved report: {name}.")
        paper = PAPER_ALL_R2_PERCENT[model]
        rows.append(dict(model=LABELS[name], r2_percent=r2, paper_r2_percent=paper,
                         difference_pp=r2-paper, n_test=int(group.n_test.sum())))
    table = pd.DataFrame(rows)

    generated = datetime.now().astimezone()
    output = Path(output_root) / generated.strftime("%Y%m%dT%H%M%S%f%z")
    output.mkdir(parents=True, exist_ok=False)
    table_dir, figure_dir = output / "table1", output / "figure3"
    table_dir.mkdir()
    figure_dir.mkdir()
    stamp = f"Generated {generated.isoformat(timespec='seconds')}"
    table.to_csv(table_dir / "table1_all.csv", index=False)
    annual.to_csv(table_dir / "annual_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 2.7), layout="constrained")
    ax.axis("off")
    cells = [[r.model, f"{r.r2_percent:.3f}", f"{r.paper_r2_percent:.2f}",
              f"{r.difference_pp:+.3f}"] for r in table.itertuples()]
    display = ax.table(cellText=cells, colLabels=["Model", "Replication (%)", "Paper (%)", "Difference (pp)"],
                       loc="center", cellLoc="center")
    display.auto_set_font_size(False)
    display.set_fontsize(11)
    display.scale(1, 1.6)
    ax.set_title("Table 1 — All stocks | 1987–2016\nPooled out-of-sample R², zero benchmark", pad=10)
    fig.text(.5, .01, stamp, ha="center", fontsize=8, color="0.4")
    for extension in ("png", "svg"):
        fig.savefig(table_dir / f"table1_all.{extension}", dpi=180)
    plt.close(fig)

    for name in ("enet", "pcr", "pls"):
        frame = complexity[name]
        column = "nonzero_slopes" if name == "enet" else "fitted_components"
        frame.to_csv(figure_dir / f"{name}_complexity.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.plot(frame.test_year, frame[column], "o-", markersize=3, linewidth=1.5, color="#246B8E")
        ax.set(title=f"Figure 3 — {LABELS[name]}", xlabel="Test year",
               ylabel="Nonzero slopes (excluding intercept)" if name == "enet" else "Number of components",
               xlim=(1986.5, 2016.5), ylim=(0, max(1, frame[column].max()) * 1.1))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks([1987, 1990, 1995, 2000, 2005, 2010, 2016])
        ax.grid(alpha=.2)
        fig.suptitle(stamp, fontsize=8, color="0.4")
        for extension in ("png", "svg"):
            fig.savefig(figure_dir / f"{name}.{extension}", dpi=180)
        plt.close(fig)
    (output / "sources.json").write_text(json.dumps(dict(
        generated_at=generated.isoformat(), sources={k: str(v.resolve()) for k, v in sources.items()},
        dataset=dataset_identity, years=YEARS,
        checks="Common observations and returns; 12 months/year; pooled R2 matches saved reports",
    ), indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        f"{stamp}\nAll-stock Table 1 only. R2 pools squared errors over 1987–2016.\n"
        "Paper references reuse core.replication_report.PAPER_ALL_R2_PERCENT.\n"
        "Figure 3 shows saved selected complexity; paper paths are not overlaid.\n"
        "OLS-3 has no tuned-complexity plot. No models were refitted.\n"
        "Inputs: imported prediction CSVs and combined complexity CSVs.\n"
        "Checks validate prediction comparability, not all estimation assumptions or checkpoint hashes.\n",
        encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Saved: {output.resolve()}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("results/scrp_imports"))
    parser.add_argument("--output-root", type=Path, default=Path("results/replication_comparison"))
    args = parser.parse_args()
    make_comparison(args.source_root, args.output_root)
