"""Table 1 (All stocks) and Figure 3 (PCR/PLS/ENet) from annual checkpoints.

No loading of the modeling panel, fitting, or resume compatibility check is needed:
reporting can also inspect results produced by older execution code.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.run_state import atomic_write, load_run_config, read_checkpoints, resolve_run_dir, run_lock


# GKX (2020), Table 1, printed p. 2250: percentage R2, 1987-2016, full universe.
PAPER_ALL_R2_PERCENT = {"OLS_HUBER": -3.46, "PLS": .27, "PCR": .26,
                        "ENET_HUBER": .11, "OLS3_HUBER": .16, "GLM_HUBER": .19, "RF": .33}


def report_tables(records, cfg):
    """Use a common OOS panel for Table 1; retain all completed complexity paths."""
    models = list(cfg.RUN.MODELS.ENABLED)
    years = list(range(cfg.RUN.SPLIT.FIRST_TEST_YEAR, cfg.RUN.SPLIT.LAST_TEST_YEAR + 1))
    common_years = [year for year in years if all((year, model) in records for model in models)]
    for year in common_years:
        reference = None
        for model in models:
            panel = pd.DataFrame(records[year, model]["predictions"])
            if (panel.empty or panel.row_position.duplicated().any() or
                    not (panel.specification == model).all() or not (panel.test_year == year).all() or
                    not np.isfinite(panel[["observed", "prediction"]].to_numpy()).all()):
                raise ValueError(f"Invalid prediction panel: {model}, {year}.")
            identity = panel[["row_position", "return_month", "observed"]].sort_values("row_position").reset_index(drop=True)
            if reference is not None and not identity.equals(reference):
                raise ValueError(f"Models do not share the same test observations in {year}.")
            reference = identity

    table = []
    for model in models:
        count, sse, denominator = 0, 0., 0.
        for year in common_years:
            panel = pd.DataFrame(records[year, model]["predictions"])
            observed, prediction = panel.observed.to_numpy(), panel.prediction.to_numpy()
            count += len(panel)
            sse += float(np.sum((observed - prediction)**2))
            denominator += float(observed @ observed)
        r2 = 1 - sse / denominator if denominator else np.nan
        table.append({
            "model": model, "sample": "All available stocks", "dataset": cfg.DATA.DATASET,
            "r2_zero": r2, "r2_percent": 100 * r2,
            "paper_full_universe_1987_2016_r2_percent": PAPER_ALL_R2_PERCENT.get(model, np.nan),
            "n_test": count,
            "completed_model_years": sum((year, model) in records for year in years),
            "common_test_years": ",".join(map(str, common_years)),
            "complete_horizon": common_years == years,
        })

    complexity, curves = [], []
    for model in (name for name in models if name in {"PCR", "PLS"}):
        grid = list(cfg.RUN.MODELS[model].COMPONENT_GRID)
        for year in years:
            row = {"model": model, "test_year": year, "status": "pending",
                   "requested_components": np.nan, "fitted_components": np.nan,
                   "validation_loss": np.nan, "runner_up_loss_gap": np.nan,
                   "at_grid_min": False, "at_grid_max": False,
                   "grid_min": min(grid), "grid_max": max(grid)}
            if (year, model) in records:
                tuning = records[year, model]["tuning"]
                candidates = tuning["validation_results"]
                scores = np.array([candidate["validation_loss"] for candidate in candidates])
                if not len(scores) or not np.isfinite(scores).all():
                    raise ValueError(f"Invalid tuning scores: {model}, {year}.")
                selected_index = int(np.argmin(scores))  # Same first-tie rule as tuning.
                selected = candidates[selected_index]
                if selected["params"] != tuning["best_params"]:
                    raise ValueError(f"Selected parameters disagree with tuning scores: {model}, {year}.")
                requested = selected["params"]["n_components"]
                actual = selected["n_components"]
                if actual is None or not 0 <= actual <= requested:
                    raise ValueError(f"Missing or invalid fitted component count: {model}, {year}.")
                row.update(status="complete", requested_components=requested,
                           fitted_components=actual, validation_loss=float(scores[selected_index]),
                           runner_up_loss_gap=float(np.sort(scores)[1] - scores[selected_index]) if len(scores) > 1 else np.nan,
                           at_grid_min=requested == min(grid), at_grid_max=requested == max(grid))
                for index, candidate in enumerate(candidates):
                    curves.append({"model": model, "test_year": year,
                                   "requested_components": candidate["params"]["n_components"],
                                   "fitted_components": candidate["n_components"],
                                   "validation_loss": candidate["validation_loss"],
                                   "loss_above_best": candidate["validation_loss"] - scores[selected_index],
                                   "selected": index == selected_index})
            complexity.append(row)
    for model in (name for name in models if name in {"ENET_HUBER", "OLS3_HUBER", "GLM_HUBER"}):
        for year in years:
            row = {"model": model, "test_year": year, "status": "pending",
                   "nonzero_slopes": np.nan, "active_groups": np.nan, "penalty_strength": np.nan,
                   "huber_delta": np.nan, "validation_loss": np.nan,
                   "at_grid_min": False, "at_grid_max": False}
            if (year, model) in records:
                tuning = records[year, model]["tuning"]
                if "coefficients" not in tuning:
                    # Earlier generic ENet runs saved predictions but no coefficient diagnostics.
                    row["status"] = "diagnostics_unavailable"
                    complexity.append(row)
                    continue
                candidates = tuning["validation_results"]
                scores = np.asarray([c["validation_loss"] for c in candidates])
                if not len(scores) or not np.isfinite(scores).all() or not all(c["converged"] for c in candidates):
                    raise ValueError(f"Invalid Huber tuning results: {model}, {year}.")
                selected = int(np.argmin(scores))
                if candidates[selected]["params"] != tuning["best_params"]:
                    raise ValueError(f"Selected parameters disagree with tuning scores: {model}, {year}.")
                count = int(np.count_nonzero(tuning["coefficients"]))
                expected_columns = len(tuning["feature_names"])
                if model == "GLM_HUBER":
                    sizes = tuning["spline_state"]["group_sizes"]
                    expected_columns = sum(sizes)
                    if len(sizes) != len(tuning["feature_names"]) or any(size not in (1, 5) for size in sizes):
                        raise ValueError("Invalid GLM spline groups.")
                    boundaries = np.r_[0, np.cumsum(sizes)]
                    coefficients = np.asarray(tuning["coefficients"])
                    active = sum(int(np.any(coefficients[a:b] != 0)) for a, b in zip(boundaries[:-1], boundaries[1:]))
                    if active != tuning["active_groups"] or not tuning["converged"]:
                        raise ValueError("Invalid GLM final group count/convergence.")
                    row["active_groups"] = active
                if count != tuning["nonzero_slopes"] or len(tuning["coefficients"]) != expected_columns:
                    raise ValueError(f"Coefficient counts disagree: {model}, {year}.")
                penalties = [c["params"]["penalty_strength"] for c in candidates]
                penalty = penalties[selected]
                row.update(status="complete", nonzero_slopes=count, penalty_strength=penalty,
                           huber_delta=tuning["huber_delta"], validation_loss=float(scores[selected]),
                           at_grid_min=penalty == min(penalties), at_grid_max=penalty == max(penalties))
                for index, candidate in enumerate(candidates):
                    curves.append({"model": model, "test_year": year,
                                   "penalty_strength": penalties[index],
                                   "nonzero_slopes": candidate["nonzero_slopes"],
                                   "active_groups": candidate.get("active_groups", np.nan),
                                   "huber_delta": candidate["huber_delta"],
                                   "iterations": candidate["iterations"], "stationarity": candidate["stationarity"],
                                   "validation_loss": candidate["validation_loss"],
                                   "loss_above_best": candidate["validation_loss"] - scores[selected],
                                   "selected": index == selected})
            complexity.append(row)
    if "RF" in models:
        for year in years:
            row = dict(model="RF", test_year=year, status="pending", average_tree_depth=np.nan,
                       max_depth=np.nan, max_features=np.nan, validation_loss=np.nan,
                       at_grid_min=False, at_grid_max=False)
            if (year, "RF") in records:
                tuning = records[year, "RF"]["tuning"]
                candidates = tuning["validation_results"]
                scores = np.asarray([c["validation_loss"] for c in candidates])
                if not len(scores) or not np.isfinite(scores).all():
                    raise ValueError(f"Invalid RF tuning scores: {year}.")
                selected = int(np.argmin(scores))
                params = candidates[selected]["params"]
                depths = np.asarray(tuning["tree_depths"])
                if (params != tuning["best_params"] or len(depths) != params["n_estimators"] or
                        not np.isfinite(depths).all() or (depths < 0).any() or
                        (depths > params["max_depth"]).any() or
                        not np.isclose(depths.mean(), tuning["average_tree_depth"])):
                    raise ValueError(f"Invalid RF selection/depth diagnostics: {year}.")
                row.update(status="complete", average_tree_depth=float(depths.mean()),
                           max_depth=params["max_depth"], max_features=params["max_features"],
                           validation_loss=float(scores[selected]),
                           at_grid_min=params["max_depth"] == min(cfg.RUN.MODELS.RF.DEPTH_GRID),
                           at_grid_max=params["max_depth"] == max(cfg.RUN.MODELS.RF.DEPTH_GRID))
                for index, candidate in enumerate(candidates):
                    curves.append(dict(model="RF", test_year=year,
                        max_depth=candidate["params"]["max_depth"], max_features=candidate["params"]["max_features"],
                        average_tree_depth=candidate["average_tree_depth"],
                        validation_loss=candidate["validation_loss"],
                        loss_above_best=candidate["validation_loss"] - scores[selected], selected=index == selected))
            complexity.append(row)
    return (pd.DataFrame(table), pd.DataFrame(complexity),
            pd.DataFrame(curves, columns=["model", "test_year", "requested_components", "fitted_components",
                                         "validation_loss", "loss_above_best", "selected", "penalty_strength",
                                         "nonzero_slopes", "active_groups", "huber_delta", "iterations", "stationarity",
                                         "max_depth", "max_features", "average_tree_depth"]))


def plot_complexity(complexity, cfg, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    models = [name for name in ("PCR", "PLS", "ENET_HUBER", "GLM_HUBER", "RF") if name in cfg.RUN.MODELS.ENABLED]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.5 * len(models)),
                             sharex=True, layout="constrained", squeeze=False)
    axes = axes[:, 0]
    for ax, model in zip(axes, models):
        group = complexity[complexity.model == model] if not complexity.empty else pd.DataFrame()
        if model == "RF":
            if not group.empty:
                ax.plot(group.test_year, group.average_tree_depth, "o-", color="#246B8E", markersize=3)
            ax.set(title="RF", ylabel="Average realized tree depth")
            ax.grid(alpha=.2)
            continue
        if model in {"ENET_HUBER", "GLM_HUBER"}:
            if not group.empty:
                values = group.active_groups if model == "GLM_HUBER" else group.nonzero_slopes
                ax.plot(group.test_year, values, "o-", color="#246B8E", markersize=3)
            ax.set_title(model, loc="left", fontweight="bold")
            ax.set_ylabel("Active input groups" if model == "GLM_HUBER" else "Nonzero slopes (excludes intercept)")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.2)
            continue
        if not group.empty:
            # Pending years remain NaN, which breaks lines instead of connecting gaps.
            ax.plot(group.test_year, group.fitted_components, "o-", color="#246B8E", markersize=3, label="Fitted components")
            different = group.requested_components != group.fitted_components
            ax.scatter(group.loc[different, "test_year"], group.loc[different, "requested_components"],
                       marker="x", color="#737373", label="Requested (if different)")
            at_max = group.at_grid_max
            ax.scatter(group.loc[at_max, "test_year"], group.loc[at_max, "fitted_components"],
                       marker="^", color="#BE5932", s=55, label="At grid maximum")
            ax.set_ylim(-.5, max(1, group.grid_max.max()) * 1.05)
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.set_title(model, loc="left", fontweight="bold")
        ax.set_ylabel("Number of components")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Test year (annual model reestimation)")
    axes[-1].set_xlim(cfg.RUN.SPLIT.FIRST_TEST_YEAR - .5, cfg.RUN.SPLIT.LAST_TEST_YEAR + .5)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"Figure 3 diagnostic: selected complexity\n{cfg.DATA.DATASET} | gaps indicate pending years", fontsize=12)
    fig.savefig(output / "figure3_complexity.png", dpi=180)
    fig.savefig(output / "figure3_complexity.svg")
    plt.close(fig)


def plot_validation_curves(curves, output):
    import matplotlib.pyplot as plt

    for year, group in curves.groupby("test_year", sort=True):
        models = [name for name in ("PCR", "PLS", "ENET_HUBER", "GLM_HUBER", "RF") if name in set(group.model)]
        if not models:
            continue  # OLS-3+H has no penalty tuning.
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.7),
                                 layout="constrained", squeeze=False)
        for ax, model in zip(axes[0], models):
            if model == "RF":
                for count, points in group[group.model == model].groupby("max_features"):
                    points = points.sort_values("max_depth")
                    ax.plot(points.max_depth, points.loss_above_best, "o-", markersize=2, label=f"{int(count)} features")
                best = group[(group.model == model) & group.selected]
                ax.scatter(best.max_depth, best.loss_above_best, color="#BE5932", zorder=3)
                ax.set(title="RF", xlabel="Maximum depth", ylabel="Validation loss above best")
                ax.legend(fontsize=6)
                ax.grid(alpha=.2)
                continue
            if model in {"ENET_HUBER", "GLM_HUBER"}:
                points = group[group.model == model].sort_values("penalty_strength")
                ax.semilogx(points.penalty_strength, points.loss_above_best, "o-")
                best = points[points.selected]
                ax.scatter(best.penalty_strength, best.loss_above_best, color="#BE5932", zorder=3)
                ax.set(title=model, xlabel="Penalty strength", ylabel="Validation loss above best")
                ax.grid(alpha=.2)
                continue
            points = group[group.model == model].sort_values("requested_components")
            ax.plot(points.requested_components, points.loss_above_best, "o-", markersize=2, color="#246B8E")
            best = points[points.selected]
            ax.scatter(best.requested_components, best.loss_above_best, color="#BE5932", zorder=3)
            ax.set(title=model, xlabel="Requested components", ylabel="Validation loss above best")
            ax.grid(alpha=.2)
        fig.suptitle(f"Validation tuning curves | test year {year}")
        fig.savefig(output / f"validation_{year}.png", dpi=160)
        plt.close(fig)


def make_report(run_dir):
    folder = resolve_run_dir(run_dir)
    with run_lock(folder):
        cfg = load_run_config(folder)
        records = read_checkpoints(folder, cfg, lazy=True)
        table, complexity, curves = report_tables(records, cfg)
        output = folder / "table1_figure3"
        output.mkdir(exist_ok=True)
        for name, frame in (("table1_all", table), ("complexity", complexity), ("tuning_curves", curves)):
            atomic_write(output / f"{name}.csv", frame.to_csv(index=False))
        annual = pd.DataFrame([record["metrics"] for record in records.values()])
        if not annual.empty:
            annual["r2_percent"] = annual.r2_zero * 100
        atomic_write(output / "annual_metrics.csv", annual.to_csv(index=False))
        atomic_write(output / "comparison_notes.txt",
                     "Table 1 uses the intersection of completed test years across all enabled models.\n"
                     "Reference values are from GKX Table 1, All row, full universe, 1987-2016.\n"
                     "Development subsamples, partial periods, and unverified settings are not exact replications.\n"
                     "Figure 3 displays PCR/PLS components or ENet nonzero slopes (intercept excluded).\n"
                     "GLM uses active input groups; RF uses average realized tree depth, not its configured cap.\n"
                     "Missing years are gaps, not zeros. OLS-3 has no tuned complexity plot.\n"
                     "Selected counts can change sharply when validation losses are nearly tied; inspect tuning curves.\n")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        atomic_write(output / "source_identity.json", json.dumps({"dataset": manifest["dataset"],
                     "config": manifest["config"], "completed_model_years": len(records)}, indent=2))
        plot_complexity(complexity, cfg, output)
        plot_validation_curves(curves, output)
    print(table.to_string(index=False))
    print(f"Saved Table 1/Figure 3 diagnostics: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Timestamped run directory (not its parent).")
    make_report(parser.parse_args().run_dir)
