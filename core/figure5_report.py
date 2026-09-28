"""Figure 5 from saved Figure 4 annual scores; never load data or refit models."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rank_importance(annual):
    """Rank within model/year, average annual ranks, then sum across models.

    Rank 1 is most important; exact ties receive average ranks. Keep signed
    scores and all predictors, including zeros. Require identical coverage.
    """
    frame = annual.loc[:, ["model", "year", "predictor", "importance"]].copy()
    if (frame.empty or frame.isna().any().any()
            or frame.duplicated(["model", "year", "predictor"]).any()
            or not np.isfinite(frame.importance).all()):
        raise ValueError("Require complete, finite, unique annual importance records.")
    expected = set(frame.predictor)
    years = set(frame.year)
    for _, records in frame.groupby("model"):
        if set(records.year) != years:
            raise ValueError("Figure 5 models must cover the same years.")
        if any(set(group.predictor) != expected for _, group in records.groupby("year")):
            raise ValueError("Figure 5 models must cover the same predictor groups.")
    frame["annual_rank"] = frame.groupby(["model", "year"]).importance.rank(
        ascending=False, method="average")
    summary = frame.groupby(["model", "predictor"], sort=False).agg(
        mean_rank=("annual_rank", "mean"), n_years=("year", "nunique")).reset_index()
    matrix = summary.pivot(index="predictor", columns="model", values="mean_rank")
    ordering = matrix.sum(axis=1).rename("rank_sum").reset_index().sort_values(
        ["rank_sum", "predictor"], kind="stable")
    matrix = matrix.loc[ordering.predictor]
    summary["overall_rank_sum"] = summary.predictor.map(matrix.sum(axis=1))
    return frame, summary, matrix


def make_report(reports, output_root=Path("results/replication_comparison")):
    sources, frames, reference = [], [], None
    for report in reports:
        folder = Path(report).resolve()
        metadata = json.loads((folder / "methodology.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise ValueError("Figure 4 source report is incomplete.")
        identity = {key: metadata[key] for key in ("dataset", "grouping", "author_grouping_verified")}
        if reference is not None and identity != reference:
            raise ValueError("Figure 4 data identities or grouping conventions differ.")
        reference = identity
        path = folder / "annual_importance.csv"
        frame = pd.read_csv(path)
        if set(frame.year) != set(metadata["years"]):
            raise ValueError("Annual score years differ from source metadata.")
        frames.append(frame)
        sources.append(dict(folder=str(folder), annual_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not frames:
        raise ValueError("Provide at least one Figure 4 report.")
    annual, summary, matrix = rank_importance(pd.concat(frames, ignore_index=True))
    generated = datetime.now().astimezone()
    output = Path(output_root) / generated.strftime("%Y%m%dT%H%M%S%f%z") / "figure5"
    output.mkdir(parents=True, exist_ok=False)
    annual.to_csv(output / "annual_ranks.csv", index=False)
    summary.to_csv(output / "mean_ranks.csv", index=False)
    matrix.to_csv(output / "rank_matrix.csv")
    metadata = dict(**reference, status="incomplete", sources=sources,
                    years=sorted(annual.year.unique().tolist()), models=list(matrix.columns),
                    n_predictors=len(matrix), generated_at=generated.isoformat(),
                    aggregation="rank within each model/year, average years, sum model mean ranks",
                    tie_policy="average ranks; alphabetical row order for tied sums",
                    negative_policy="rank signed raw scores without clipping",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata_path = output / "methodology.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(matrix.columns) * .85), max(5, len(matrix) * .18)),
                           layout="constrained")
    # Low mean rank is best and darkest. Use a common rank scale across models.
    heatmap = ax.imshow(matrix.to_numpy(), cmap="Blues_r", vmin=1, vmax=max(2, len(matrix)), aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_title(f"Figure 5 diagnostic | {min(metadata['years'])}–{max(metadata['years'])}\n"
                 f"{len(matrix)} groups; {len(matrix.columns)} models; author grouping unverified")
    fig.colorbar(heatmap, ax=ax, label="Mean annual rank (1 = most important)", shrink=.5)
    for extension in ("png", "svg"):
        fig.savefig(output / f"characteristic_ranks.{extension}", dpi=180)
    plt.close(fig)
    metadata["status"] = "complete"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Figure 5 diagnostics: {output.resolve()}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure4", type=Path, nargs="+", required=True,
                        help="Completed Figure 4 directories containing annual_importance.csv.")
    parser.add_argument("--output-root", type=Path, default=Path("results/replication_comparison"))
    args = parser.parse_args()
    make_report(args.figure4, args.output_root)
