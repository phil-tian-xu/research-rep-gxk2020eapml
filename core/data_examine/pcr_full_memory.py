"""Run the simple full-array PCR experiment on the full GKX universe.

python -B -m core.data_examine.pcr_full_memory

Defaults: train 1957-1974, validate 1975-1986, test 1987, fixed K=10.
Uses the same original dense PCR fit and memory reporting as pcr_memory.
"""

import argparse

from core.data_examine.pcr_memory import experiment
from utils.path_info import data_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-year", type=int, default=1987)
    parser.add_argument("--components", type=int, default=10,
                        help="Fixed PCR K; no hyperparameter search.")
    parser.add_argument("--output-dir", default="results/pcr_full_memory")
    args = parser.parse_args(argv)
    parquet_dir = data_path / "processed" / "gkx_parquet"
    print(f"Full-universe Parquet source: {parquet_dir}", flush=True)
    return experiment(parquet_dir, test_year=args.test_year,
                      components=args.components, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
