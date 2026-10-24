#!/usr/bin/env bash
#SBATCH --job-name=gkx-ols3-thresholds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-ols3-thresholds-%j.out

# Submit from the AI Hub project root. Initial resource estimates; no GPU.
# Override account/partition/QoS with your confirmed site settings if needed.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_CPUS_PER_TASK:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Use sbatch run_ols3_thresholds_aihub.sh [experiment options]; no arrays." >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
python_bin="$PWD/.venv/bin/python"
time_bin="$HOME/.local/bin/time"
if [[ ! -f core/data_examine/ols3_thresholds.py || ! -x "$python_bin" || ! -x "$time_bin" ]]; then
    echo "Required: project root, experiment module, .venv/bin/python and GNU time at $time_bin." >&2
    exit 2
fi
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export PYTHONUTF8=1
job_dir="$PWD/results/huber_threshold_logs/aihub-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$job_dir"
cp -- "$0" "$job_dir/submitted_script.sh"
printf 'Job log: %s/run.log\n' "$job_dir"
exec >> "$job_dir/run.log" 2>&1
echo "Started: $(date -Is); host=$(hostname); job=$SLURM_JOB_ID"
echo "CPUs=$SLURM_CPUS_PER_TASK; memory=${SLURM_MEM_PER_NODE:-unknown} MiB"
printf 'Command:'
printf ' %q' "$python_bin" -u -B -m core.data_examine.ols3_thresholds "$@"
printf '\n'
"$python_bin" -B -c 'import numpy; numpy.show_config()' > "$job_dir/numpy_config.txt"
"$python_bin" -B -c 'from importlib.metadata import distributions; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in distributions())))' > "$job_dir/packages.txt"
exec "$time_bin" -v -o "$job_dir/time.txt" \
    "$python_bin" -u -B -m core.data_examine.ols3_thresholds "$@"
