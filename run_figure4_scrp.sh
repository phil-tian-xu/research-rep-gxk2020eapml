#!/usr/bin/env bash
#SBATCH --job-name=gkx-figure4-scrp
#SBATCH --account=pg
#SBATCH --partition=scrp
#SBATCH --qos=c32g4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-figure4-%j.out

# Submit from the project root; all arguments go to core.figure4_report.
# Example: sbatch run_figure4_scrp.sh --experiment /absolute/path/to/experiment
# Initial resource requests, not measured cluster requirements. No GPU needed.
set -euo pipefail
if [[ $# == 0 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_CPUS_PER_TASK:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Usage: sbatch run_figure4_scrp.sh --experiment PATH [report options]; no job arrays." >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
python_bin="$PWD/.venv/bin/python"
if [[ ! -f core/figure4_report.py || ! -f core/figure4_importance.py || ! -x "$python_bin" || ! -x /usr/bin/time ]]; then
    echo "Required: project root, Figure 4 modules, .venv/bin/python and GNU /usr/bin/time." >&2
    exit 2
fi
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export PYTHONUTF8=1
job_dir="$PWD/results/figure4_logs/scrp-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$job_dir"
cp -- "$0" "$job_dir/submitted_script.sh"
printf 'Figure 4 job log: %s/run.log\n' "$job_dir"
exec >> "$job_dir/run.log" 2>&1
echo "Started: $(date -Is); host=$(hostname); job=$SLURM_JOB_ID"
echo "CPUs=$SLURM_CPUS_PER_TASK; memory=${SLURM_MEM_PER_NODE:-unknown} MiB"
printf 'Command:'
printf ' %q' "$python_bin" -u -B -m core.figure4_report "$@"
printf '\n'
"$python_bin" -V
"$python_bin" -B -c 'from importlib.metadata import distributions; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in distributions())))' > "$job_dir/packages.txt"
"$python_bin" -B -c 'import numpy; numpy.show_config()' > "$job_dir/numpy_config.txt"
sha256sum core/figure4_report.py core/figure4_importance.py > "$job_dir/report_sources.sha256"
# One process handles the selected years and writes their combined Figure 4.
# Exit status propagates to Slurm; failed/incomplete reports remain labelled incomplete.
exec /usr/bin/time -v -o "$job_dir/time.txt" \
    "$python_bin" -u -B -m core.figure4_report "$@"
