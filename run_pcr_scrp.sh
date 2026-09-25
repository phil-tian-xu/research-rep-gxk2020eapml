#!/usr/bin/env bash
#SBATCH --job-name=gkx-pcr
#SBATCH --account=pg
#SBATCH --partition=scrp
#SBATCH --qos=c32g4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out

# Submit from ~/research-rep-gxk2020eapml on the SCRP login node.
# New run: sbatch run_pcr_scrp.sh --stop-after-year 1996
# Resume:  sbatch run_pcr_scrp.sh --resume /absolute/run/directory --stop-after-year 2006
# With no stop year, run through the configured final year (2016).
set -euo pipefail

resume_dir=""
stop_year=""
while (( $# )); do
    case "$1" in
        --resume)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                echo "--resume requires a run directory" >&2; exit 2
            fi
            resume_dir="$2"
            shift 2
            ;;
        --stop-after-year)
            if (( $# < 2 )) || [[ ! "$2" =~ ^[0-9]{4}$ ]]; then
                echo "--stop-after-year requires a four-digit year" >&2; exit 2
            fi
            stop_year="$2"
            if (( stop_year < 1987 || stop_year > 2016 )); then
                echo "Stop year must be within 1987-2016" >&2; exit 2
            fi
            shift 2
            ;;
        --help|-h)
            echo "sbatch run_pcr_scrp.sh [--resume RUN_DIRECTORY] [--stop-after-year YEAR]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "Submit this script with sbatch; do not run fitting on the login node." >&2
    exit 2
fi

cd "$HOME/research-rep-gxk2020eapml"
python_bin="$PWD/.venv/bin/python"
if [[ ! -x "$python_bin" || ! -x /usr/bin/time ]]; then
    echo "Required: project .venv/bin/python and GNU /usr/bin/time." >&2
    exit 2
fi
if [[ ! -f data/processed/gkx_parquet/metadata.json ]]; then
    echo "Missing data/processed/gkx_parquet/metadata.json; check the repository data link." >&2
    exit 2
fi

args=(--config gkx_pcr_scrp)
if [[ -n "$resume_dir" ]]; then
    if [[ ! -f "$resume_dir/manifest.json" ]]; then
        echo "No run manifest in: $resume_dir" >&2; exit 2
    fi
    args=(--resume "$resume_dir")
fi
if [[ -n "$stop_year" ]]; then
    args+=(--stop-after-year "$stop_year")
fi

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

job_dir="/data/users/s1155269217/GKX2020/results/slurm/pcr-${SLURM_JOB_ID}"
mkdir -p "$job_dir"
echo "PCR job logs: $job_dir/run.log"
echo "Peak Linux resident memory and timing: $job_dir/time.txt"
cp -- "$0" "$job_dir/submitted_script.sh"
exec >> "$job_dir/run.log" 2>&1

echo "Started: $(date -Is); job=$SLURM_JOB_ID; node=$(hostname)"
echo "CPUs=$SLURM_CPUS_PER_TASK; allocated memory per node=${SLURM_MEM_PER_NODE:-unknown} MiB"
echo "Repository: $PWD"
printf 'Command: %q -u -B main.py' "$python_bin"
printf ' %q' "${args[@]}"
printf '\n'
"$python_bin" -V
"$python_bin" -c 'import numpy, scipy, pandas, pyarrow, matplotlib, yaml, iopath; from importlib.metadata import distributions; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in distributions())))' > "$job_dir/packages.txt"

# GNU time records the process peak on Linux without changing model fitting.
# Do not use the current Windows-only --memory option here.
# exec preserves Python/time's exit status for Slurm, including failures.
exec /usr/bin/time -v -o "$job_dir/time.txt" "$python_bin" -u -B main.py "${args[@]}"
