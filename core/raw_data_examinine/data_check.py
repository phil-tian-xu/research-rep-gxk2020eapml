from pathlib import Path

import pandas as pd

from utils.path_info import data_path


def main(input_files, output_dir, chunk_size=100_000):
    for filename in input_files:
        input_path = data_path / "raw" / filename
        source_output_dir = Path(output_dir) / input_path.stem

        split_monthly_csv(input_path, source_output_dir, chunk_size, filename.replace(".csv", ""))


def split_monthly_csv(input_path, output_dir, chunk_size, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    written_months = set()

    # Read as text to preserve original values and empty cells.
    with pd.read_csv(
        input_path,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
    ) as reader:
        for chunk in reader:
            months = chunk["DATE"].str[:6]

            for month, monthly_data in chunk.groupby(months):
                first_write = month not in written_months

                monthly_data.to_csv(
                    output_dir / f"{filename}_{month}.csv",
                    mode="w" if first_write else "a",
                    header=first_write,
                    index=False,
                )

                written_months.add(month)

    print(f"{input_path.name}: saved {len(written_months)} monthly files.")


if __name__ == "__main__":
    green_target_file = "green_raw_v1_19570101_20211231.csv"  # green_raw_v1_19570101_20211231, green_raw_v1_20000101_20091231, RPSdata_RFS
    xiu_target_file = "datashare.csv"
    input_files = [green_target_file] # , xiu_target_file]
    output_dir = data_path / "monthly_csv"
    chunk_size = 100_000

    main(input_files, output_dir, chunk_size)
