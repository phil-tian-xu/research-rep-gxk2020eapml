from pathlib import Path

import pandas as pd

from utils.path_info import data_path


def main(green_target_file, xiu_target_file, input_dir, output_dir):
    input_dir = data_path / "monthly_csv"
    output_dir = data_path / "monthly_csv_reshaped"

    features = """
        absacc acc aeavol age agr baspread beta betasq bm bm_ia
        cash cashdebt cashpr cfp cfp_ia chatoia chcsho chempia
        chinv chmom chpmia chtx cinvest convind currat depr
        divi divo dolvol dy ear egr ep gma grCAPX grltnoa
        herf hire idiovol ill indmom invest lev lgr maxret
        mom12m mom1m mom36m mom6m ms mve_ia mvel1 nincr
        operprof orgcap pchcapx_ia pchcurrat pchdepr
        pchgm_pchsale pchquick pchsale_pchinvt pchsale_pchrect
        pchsale_pchxsga pchsaleinv pctacc pricedelay ps quick
        rd rd_mve rd_sale realestate retvol roaq roavol roeq
        roic rsup salecash saleinv salerec secured securedind
        sgr sin sp std_dolvol std_turn stdacc stdcf tang tb
        turn zerotrade
    """.split()

    rename_maps = {
        Path(xiu_target_file).stem: {
            "DATE": "date",
            "grcapx": "grCAPX",
        },
        Path(green_target_file).stem: {
            "DATE": "date",
            "BETA": "beta",
            "grcapx": "grCAPX",
            "mve_m": "mvel1",
        },
    }

    columns = ["date", "permno"] + features

    for source, rename_map in rename_maps.items():
        source_input_dir = input_dir / source
        source_output_dir = output_dir / source

        monthly_files = sorted(source_input_dir.glob("*.csv"))
        if not monthly_files:
            raise FileNotFoundError(
                f"No monthly CSV files found in {source_input_dir}"
            )

        source_output_dir.mkdir(parents=True, exist_ok=True)

        for input_path in monthly_files:
            reshape_monthly_csv(
                input_path=input_path,
                output_path=source_output_dir / input_path.name,
                rename_map=rename_map,
                columns=columns,
            )

        print(f"{source}: reshaped {len(monthly_files)} monthly files.")


def reshape_monthly_csv(input_path, output_path, rename_map, columns):
    # Preserve cell values, empty cells, and original date formatting.
    frame = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
    )

    frame = frame.rename(columns=rename_map)

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{input_path.name}: missing columns {missing}")

    frame.loc[:, columns].to_csv(output_path, index=False)


if __name__ == "__main__":
    green_target_file = "green_raw_v1_19570101_20211231.csv"  # green_raw_v1_19570101_20211231, green_raw_v1_20000101_20091231, RPSdata_RFS
    xiu_target_file = "datashare.csv"
    input_dir = data_path / "monthly_csv"
    output_dir = data_path / "monthly_csv_reshaped"

    main(green_target_file, xiu_target_file, input_dir, output_dir)
