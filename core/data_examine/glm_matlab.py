"""Small GKX simulation GLM reference comparison; not an empirical estimator.

Reference: https://github.com/xiubooth/ML_Codes/tree/master/Simu_Matlab
See docs/working_notes/GLM MATLAB Comparison.md for conventions and commands.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import scipy
from scipy.io import loadmat, savemat


def spline_basis(X, knots):
    """cut_knots_degree2: center EACH supplied sample, including evaluation data."""
    blocks = []
    for column, anchors in zip(np.asarray(X).T, knots.T):
        x = np.where(np.isnan(column), 0., column)
        block = np.column_stack([x, (x - anchors[0])**2] +
                                [np.maximum(x - k, 0.)**2 for k in anchors[1:]])
        blocks.append(block - block.mean(axis=0))
    return np.column_stack(blocks)


def block_threshold(values, threshold, width=5):
    """Unweighted L2 block soft threshold, matching soft_threshodg."""
    blocks = np.asarray(values).reshape(-1, width)
    norms = np.sqrt(np.sum(blocks**2, axis=1))
    result = np.zeros_like(blocks)
    active = norms > threshold
    result[active] = blocks[active] - threshold * blocks[active] / norms[active, None]
    return result.ravel()


def reference_fit(X, y, strength, lipschitz, tol, *, initial=None, delta=None):
    """Authors' acceleration and iterate-change stop; returns extrapolated v.

    Squared objective is .5*sum(residual**2) + lambda*sum(group norms).
    Robust objective replaces the squared term with summed half-Huber loss.
    This faithfully bounded reference solver is NOT a convergence-certified fit.
    """
    w = np.zeros(X.shape[1]) if initial is None else initial.copy()
    v = w.copy()
    gram, cross = X.T @ X, X.T @ y
    limit = 30000 if delta is None else 3000
    stopped = False
    for t in range(limit):
        previous_v, previous_w = v, w
        gradient = gram @ v - cross if delta is None else X.T @ np.clip(X @ v - y, -delta, delta)
        w = block_threshold(v - gradient / lipschitz, strength / lipschitz)
        v = w + t / (t + 3) * (w - previous_w)
        difference = v - previous_v
        if np.sum(difference**2) < np.sum(previous_v**2) * tol or np.sum(abs(difference)) == 0:
            stopped = True
            break
    return v, {"iterations": t + 1, "iterate_stop_met": stopped}


def make_fixture():
    """One shared MAT file avoids different MATLAB/NumPy random generators."""
    rng = np.random.default_rng(20260927)
    fixture = {}
    for split, n, shift in [("train", 240, 0.), ("validation", 100, .15), ("test", 120, -.1)]:
        X = rng.uniform(-1, 1, (n, 4)) + shift
        X[:, 2] = np.round(X[:, 2], 1)  # ties in one continuous input
        X[:, 3] = 2.  # constant input tests zero-scale handling
        y = .04 + .45 * X[:, 0] + .3 * X[:, 1]**2 + rng.normal(0, .06, n)
        y[::19] += 1.  # exercise the Huber tail
        fixture[f"X_{split}"] = X
        fixture[f"y_{split}"] = y
    fixture.update(lambdas=10.**np.linspace(.5, 3., 26), delta=.2, tol=1e-10,
                   knot_probabilities=np.array([0., .25, .5, .75]),
                   probe_values=np.array([3., 4., 0., 0., 0., .1, 0., 0., 0., 0.]),
                   probe_threshold=1.)
    return fixture


def calculate(fixture):
    """Run the GLM section of MC_Reg on a small, fully observed fixture."""
    f = fixture
    raw_scale = np.std(f["X_train"], axis=0, ddof=1)
    raw_scale = np.where(raw_scale > 0, raw_scale, 1.)
    X_train = f["X_train"] / raw_scale
    # MATLAB default quantile places sorted observations at (i-.5)/n (Hazen).
    knots = np.quantile(X_train, f["knot_probabilities"], axis=0, method="hazen")
    basis = {split: spline_basis(f[f"X_{split}"] / raw_scale, knots)
             for split in ("train", "validation", "test")}
    scale = np.std(basis["train"], axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.)
    Z = {split: matrix / scale for split, matrix in basis.items()}
    y_mean = np.mean(f["y_train"])
    y = f["y_train"] - y_mean
    lipschitz = np.linalg.svd(Z["train"].T @ Z["train"], compute_uv=False)[0]
    paths, diagnostics = [], []
    for strength in f["lambdas"]:
        beta, info = reference_fit(Z["train"], y, strength, lipschitz, f["tol"])
        paths.append(beta)
        diagnostics.append(info)
    paths = np.column_stack(paths)
    validation_predictions = Z["validation"] @ paths + y_mean
    validation_r2 = 1 - np.sum((validation_predictions - f["y_validation"][:, None])**2, axis=0) / np.sum((f["y_validation"] - y_mean)**2)
    selected = int(np.argmax(validation_r2))  # first maximum, as fw1.m
    beta = paths[:, selected]
    robust, robust_info = reference_fit(Z["train"], y, f["lambdas"][selected], lipschitz,
                                       f["tol"], initial=beta, delta=f["delta"])
    result = dict(raw_scale=raw_scale, knots=knots, basis_scale=scale,
                  target_mean=y_mean, lipschitz=lipschitz, coefficient_path=paths,
                  validation_r2_path=validation_r2, selected_index=selected + 1,
                  selected_lambda=f["lambdas"][selected], beta_squared=beta, beta_huber=robust,
                  prox_probe=block_threshold(f["probe_values"], f["probe_threshold"]))
    for split in Z:
        result[f"basis_{split}"] = basis[split]
        result[f"Z_{split}"] = Z[split]
    for name, coefficients in [("squared", beta), ("huber", robust)]:
        norms = np.linalg.norm(coefficients.reshape(-1, 5), axis=1)
        residual = Z["train"] @ coefficients - y
        absolute = abs(residual)
        half_loss = (.5 * residual**2 if name == "squared" else
                     np.where(absolute <= f["delta"], .5 * residual**2,
                              f["delta"] * absolute - .5 * f["delta"]**2))
        gradient = Z["train"].T @ (residual if name == "squared" else np.clip(residual, -f["delta"], f["delta"]))
        mapping = lipschitz * (coefficients - block_threshold(coefficients - gradient / lipschitz, result["selected_lambda"] / lipschitz))
        result[f"objective_{name}"] = half_loss.sum() + result["selected_lambda"] * norms.sum()
        result[f"stationarity_{name}"] = np.max(abs(mapping))
        result[f"group_norms_{name}"] = norms
        for split in Z:
            prediction = Z[split] @ coefficients + y_mean
            result[f"prediction_{name}_{split}"] = prediction
            result[f"r2_{name}_{split}"] = 1 - np.sum((prediction - f[f"y_{split}"])**2) / np.sum((f[f"y_{split}"] - y_mean)**2)
    return result, {"squared_candidates": diagnostics, "huber": robust_info}


def read_mat(path):
    return {key: value for key, value in loadmat(path, simplify_cells=True).items()
            if not key.startswith("__")}


def compare_arrays(expected, actual, *, exact_values=False):
    """Finite values and matching shapes required; missing fields never pass."""
    rows = []
    exact = {"selected_index", "selected_lambda"}
    for key, expected_value in expected.items():
        a = np.asarray(expected_value).squeeze()
        b = np.asarray(actual.get(key, [])).squeeze()
        iterative = key.startswith(("beta_", "coefficient_", "prediction_", "r2_", "validation_r2", "objective_", "stationarity_", "group_norms_"))
        atol, rtol = ((1e-7, 1e-5) if iterative else (1e-10, 1e-10))
        if key in exact or exact_values:
            atol, rtol = 0., 0.
        valid = key in actual and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(abs(a - b))) if valid else None
        passed = bool(valid and np.allclose(a, b, atol=atol, rtol=rtol))
        rows.append(dict(field=key, passed=passed, max_absolute_error=error, atol=atol, rtol=rtol))
    return rows


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    fixture = make_fixture()
    result, diagnostics = calculate(fixture)
    savemat(output / "fixture.mat", fixture, oned_as="column")
    savemat(output / "python_results.mat", result, oned_as="column")
    # Keep the exact comparison implementation alongside its numerical outputs.
    shutil.copy2(__file__, output / "glm_matlab.py")
    shutil.copy2(Path(__file__).with_name("run_glm_reference.m"), output / "run_glm_reference.m")
    manifest = dict(status="awaiting_matlab", python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, diagnostics=diagnostics,
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared comparison: {output.resolve()}")
    print(f"Python selected lambda={result['selected_lambda']:.8g}; MATLAB comparison pending.")
    if not all(d["iterate_stop_met"] for d in diagnostics["squared_candidates"] + [diagnostics["huber"]]):
        print("Some reference fits reached their iteration limit; see manifest.json.")


def compare(output):
    reference = output / "matlab_results.mat"
    if not reference.exists():
        raise FileNotFoundError(f"MATLAB has not produced {reference}. Run run_glm_reference first.")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, checksum in manifest["sha256"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Comparison artifact changed: {name}. Prepare a fresh comparison.")
    actual = read_mat(reference)
    expected = read_mat(output / "python_results.mat")
    rows = compare_arrays(expected, actual)
    fixture_rows = compare_arrays(read_mat(output / "fixture.mat"), actual.get("fixture", {}), exact_values=True)
    for row in fixture_rows:
        row["field"] = "fixture." + row["field"]
    rows.extend(fixture_rows)
    sources = actual.get("author_sources", {})
    required = {"cut_knots_degree2", "proximal", "proximalH", "f_grad", "f_gradh", "soft_threshodg", "lossh", "fw1"}
    provenance_ok = required.issubset(sources) and all(isinstance(sources[k], str) and sources[k] for k in required)
    passed = all(row["passed"] for row in rows) and provenance_ok
    report = dict(passed=passed, matlab_version=str(actual.get("matlab_version", "unknown")),
                  provenance_present=provenance_ok,
                  author_source_text_sha256={k: hashlib.sha256(str(v).encode("utf-8")).hexdigest() for k, v in sources.items()},
                  matlab_output_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), checks=rows,
                  note="Numerical reference agreement only; not approval of an empirical specification or proof of convergence.")
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"MATLAB comparison: {'PASS' if passed else 'FAIL'}; report: {output / 'comparison.json'}")
    for row in rows:
        if not row["passed"]:
            print(f"  {row['field']}: max absolute error={row['max_absolute_error']}")
    if not provenance_ok:
        print("  Missing author-source provenance; use the supplied MATLAB driver.")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare", help="Generate shared synthetic inputs and Python reference results.")
    create.add_argument("--output", type=Path, default=None)
    check = sub.add_parser("compare", help="Compare saved MATLAB outputs with Python.")
    check.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        output = args.output or Path("results/glm_matlab") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prepare(output)
    elif not compare(args.run_dir):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
