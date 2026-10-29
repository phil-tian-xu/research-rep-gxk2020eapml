"""Load prepared modeling panels by dataset name; never generate data implicitly."""

import csv
import hashlib
import json

import pandas as pd

from core.model_inputs import validate_model_inputs
from utils.path_info import data_path


DATASET_PATHS = {
    "synthetic_linear_v1": data_path / "experimental" / "synthetic_linear_v1",
    "synthetic_nonlinear_v1": data_path / "experimental" / "synthetic_nonlinear_v1",
    # Prepared-panel destination only; no GKX dataset has been constructed here yet.
    "gkx": data_path / "processed" / "gkx",
    "gkx_dev_p05_seed2020_parquet": data_path / "processed" / "gkx_dev_p05_seed2020_parquet",
    "gkx_parquet": data_path / "processed" / "gkx_parquet",
    "gkx_dev_p05_seed2020": data_path / "processed" / "gkx_dev_p05_seed2020",
    "gkx_dev_p10_seed2020": data_path / "processed" / "gkx_dev_p10_seed2020",
    "gkx_compact_full": data_path / "processed" / "gkx_compact_full",
    "gkx_compact_p05": data_path / "processed" / "gkx_compact_p05",
    "gkx_expanded_full": data_path / "processed" / "gkx_expanded_full",
    "gkx_expanded_p05": data_path / "processed" / "gkx_expanded_p05",
}


def validate_feature_dataset(metadata):
    """Reject stale/mislabeled compact or expanded manifests before fitting."""
    name = metadata.get("dataset_name", "")
    if not isinstance(name, str) or not name.startswith(("gkx_compact_", "gkx_expanded_")):
        return  # Original datasets keep their existing metadata contract.
    kind = "compact" if name.startswith("gkx_compact_") else "expanded"
    spec = metadata.get("feature_specification")
    if not isinstance(spec, dict) or spec.get("kind") != kind:
        raise ValueError(f"{name}: missing/mismatched feature specification; prepare this dataset again.")
    chars, industries, macros, features = (spec.get("characteristics"), spec.get("industry_columns"),
                                          spec.get("macro_columns"), metadata.get("feature_columns"))
    for columns in (chars, industries, macros, features):
        if (not isinstance(columns, list) or any(not isinstance(c, str) for c in columns) or
                len(columns) != len(set(columns))):
            raise ValueError(f"{name}: feature specification must declare unique column-name lists.")
    if len(chars) != 94 or any(not c.startswith("sic2_") for c in industries):
        raise ValueError(f"{name}: expected 94 characteristics and SIC industry indicators.")
    macro_variables = ("dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar")
    if kind == "compact":
        expected_macros = [f"macro_{m}" for m in macro_variables]
        expected = chars + expected_macros + industries
        valid = macros == expected_macros and features == expected
    else:
        expected = chars + [f"{c}__macro_{m}" for m in macro_variables for c in chars] + industries
        valid = not macros and len(features) == len(expected) and set(features) == set(expected)
    if not valid or len(expected) != len(set(expected)):
        raise ValueError(f"{name}: predictors do not match the {kind} feature definition; prepare this dataset again.")


def load_dataset(name):
    """Return X, y, return_month, dataset_info from panel.csv and metadata.json.

    Metadata declares column roles/order, decimal target units, row count, and
    panel SHA-256. Identifiers/dates/targets cannot enter the predictor list.
    Dataset info records provenance for saved run results. Row positions
    preserve the panel's order so predictions can be joined back to identifiers.
    """
    if name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset {name!r}; available names: {list(DATASET_PATHS)}.")
    folder = DATASET_PATHS[name]
    panel_path, metadata_path = folder / "panel.csv", folder / "metadata.json"
    if not panel_path.is_file() or not metadata_path.is_file():
        instruction = ("Run python -B -m core.data_query.generate_experimental_data first."
                       if name.startswith("synthetic_") else "Prepare the modeling panel and metadata first.")
        raise FileNotFoundError(f"Dataset {name!r} is not ready at {folder}. {instruction}")
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("schema_version") != 1 or metadata.get("dataset_name") != name:
        raise ValueError("Dataset metadata must have schema_version=1 and the selected dataset_name.")
    if metadata.get("return_units") != "decimal":
        raise ValueError("Prepared dataset targets must be excess returns in decimal units.")
    with panel_path.open("rb") as source:
        actual_digest = hashlib.file_digest(source, "sha256").hexdigest()
    if actual_digest != metadata.get("panel_sha256"):
        raise ValueError("Panel SHA-256 does not match metadata; dataset contents have changed.")
    features = metadata["feature_columns"]
    identifiers = metadata["id_columns"]
    target, date = metadata["target_column"], metadata["return_month_column"]
    feature_date = metadata["feature_month_column"]
    if not isinstance(features, list) or not features or not isinstance(identifiers, list) or not identifiers:
        raise ValueError("feature_columns and id_columns must be nonempty lists.")
    columns = identifiers + [feature_date, date, target] + features
    if not all(isinstance(column, str) for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("Metadata column roles must be unique strings, with no overlap.")
    with panel_path.open(encoding="utf-8", newline="") as source:
        header = next(csv.reader(source), [])
    if len(set(header)) != len(header) or not set(columns).issubset(header):
        raise ValueError("Panel has duplicate column names or is missing declared columns.")
    panel = pd.read_csv(panel_path, usecols=columns,
                        dtype={column: "string" for column in identifiers + [date, feature_date]},
                        float_precision="round_trip")
    if len(panel) != metadata["n_rows"]:
        raise ValueError("Panel row count does not match metadata.")
    if panel[identifiers + [date, feature_date]].isna().any().any():
        raise ValueError("Identifiers and month labels must not be missing.")
    return_month = pd.PeriodIndex(panel[date], freq="M")
    feature_month = pd.PeriodIndex(panel[feature_date], freq="M")
    if return_month.isna().any() or feature_month.isna().any() or not ((feature_month + 1) == return_month).all():
        raise ValueError("Each feature_month must be exactly one month before return_month.")
    # Normalize first so alternate textual labels for the same month cannot hide duplicates.
    panel[date] = return_month.astype(str)
    if panel.duplicated(identifiers + [date]).any():
        raise ValueError("Duplicate asset/return-month observations in modeling panel.")
    X, y = panel.loc[:, features], panel[target]
    validate_model_inputs(X, y)  # Shared shape/numeric/finite checks; no fitting or preprocessing.
    dataset_info = {
        "name": name, "directory": str(folder.resolve()), "panel_sha256": actual_digest,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(), "metadata": metadata,
    }
    return X, y, return_month, dataset_info
