"""Generate saved toy panels separately from model experiments.

Run: python -B -m core.data_query.generate_experimental_data
Creates linear and nonlinear panels in data/experimental/. Existing datasets
are never overwritten. These are software test data, not the paper's simulations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import data_path


COEFFICIENTS = (0.015, -0.01, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0)


def generate_panel(
    *, kind="linear", seed=2020, start_month="1957-03", end_month="1988-12",
    stocks_per_month=8, coefficients=COEFFICIENTS, intercept=0.002, noise_std=0.05,
):
    """Return a panel and generator settings, without reading config or writing files.

    x columns represent information at feature_month, one month before the target.
    Separate random streams keep existing rows stable if the end month is extended.
    """
    if kind not in {"linear", "nonlinear"}:
        raise ValueError("kind must be 'linear' or 'nonlinear'.")
    if isinstance(stocks_per_month, bool) or not isinstance(stocks_per_month, int) or stocks_per_month < 1:
        raise ValueError("stocks_per_month must be a positive integer.")
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or not coefficients.size or not np.isfinite(coefficients).all():
        raise ValueError("coefficients must be a nonempty finite vector.")
    if kind == "nonlinear" and len(coefficients) < 3:
        raise ValueError("The nonlinear generator requires at least three predictors.")
    if not np.isfinite(intercept) or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("intercept and noise_std must be finite; noise_std must be nonnegative.")
    start, end = pd.Period(start_month, freq="M"), pd.Period(end_month, freq="M")
    if start > end:
        raise ValueError("start_month must not follow end_month.")
    months = pd.period_range(start, end, freq="M").repeat(stocks_per_month)
    feature_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    features = np.random.default_rng(feature_seed).normal(size=(len(months), len(coefficients)))
    noise = np.random.default_rng(noise_seed).normal(scale=noise_std, size=len(months))
    signal = intercept + features @ coefficients
    nonlinear_terms = None
    if kind == "nonlinear":
        signal += 0.02 * (features[:, 0]**2 - 1.0) + 0.015 * features[:, 1] * features[:, 2]
        nonlinear_terms = {"centered_x0_squared": 0.02, "x1_times_x2": 0.015}
    panel = pd.DataFrame(features, columns=[f"x{j}" for j in range(len(coefficients))])
    panel.insert(0, "asset_id", np.tile(np.arange(1, stocks_per_month + 1), len(months) // stocks_per_month))
    panel.insert(1, "feature_month", (months - 1).astype(str))
    panel.insert(2, "return_month", months.astype(str))
    panel["excess_return"] = signal + noise
    settings = {
        "kind": kind, "seed": seed, "start_month": str(start), "end_month": str(end),
        "stocks_per_month": stocks_per_month, "coefficients": coefficients.tolist(),
        "intercept": intercept, "noise_std": noise_std, "nonlinear_terms": nonlinear_terms,
        "feature_distribution": "independent standard normal",
        "noise_distribution": "independent normal with configured standard deviation",
        "random_streams": "SeedSequence(seed).spawn(2): predictors, then noise",
        "numpy_version": np.__version__,
    }
    return panel, settings


def save_generated_dataset(output_dir, *, dataset_name, **generator_settings):
    """Save panel.csv and metadata.json in a new directory; reject overwrites."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Dataset already exists: {output_dir}. Existing data will not be overwritten.")
    panel, settings = generate_panel(**generator_settings)
    output_dir.mkdir(parents=True, exist_ok=False)
    panel_path = output_dir / "panel.csv"
    panel.to_csv(panel_path, index=False, float_format="%.17g", lineterminator="\n")
    with panel_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {
        "schema_version": 1, "dataset_name": dataset_name, "dataset_type": "synthetic",
        "description": "Toy software-validation panel; not an empirical or Monte Carlo replication of GKX.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "core.data_query.generate_experimental_data", "generation": settings,
        "id_columns": ["asset_id"], "feature_month_column": "feature_month",
        "return_month_column": "return_month", "target_column": "excess_return",
        "feature_columns": [f"x{j}" for j in range(len(settings["coefficients"]))],
        "return_units": "decimal", "n_rows": len(panel), "panel_sha256": digest,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["linear", "nonlinear", "all"], default="all")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--start-month", default="1957-03")
    parser.add_argument("--end-month", default="1988-12")
    parser.add_argument("--stocks-per-month", type=int, default=8)
    args = parser.parse_args()
    kinds = ("linear", "nonlinear") if args.dataset == "all" else (args.dataset,)
    for kind in kinds:
        name = f"synthetic_{kind}_v1"
        folder = save_generated_dataset(
            data_path / "experimental" / name, dataset_name=name, kind=kind,
            seed=args.seed, start_month=args.start_month, end_month=args.end_month,
            stocks_per_month=args.stocks_per_month,
        )
        print(f"Generated {name}: {folder}")


if __name__ == "__main__":
    main()
