#!/usr/bin/env bash
#SBATCH --job-name=gkx-aihub
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0%1
#SBATCH --output=slurm-%A_%a.out

# Use run_experiment.py --aihub to set the array and record the submission.
# Account, partition and QoS use site defaults unless supplied to the launcher.
# This CPU workflow does not request a GPU.
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "Usage: sbatch run_annual_aihub.sh EXPERIMENT_DIRECTORY" >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
experiment="$1"
if [[ "$experiment" != /* || ! -f "$experiment/experiment.json" ]]; then
    echo "Provide the absolute path of a prepared experiment." >&2
    exit 2
fi
python_bin="$PWD/.venv/bin/python"
if [[ ! -x "$python_bin" || ! -x /usr/bin/time ]]; then
    echo "Required: project .venv/bin/python and GNU /usr/bin/time." >&2
    exit 2
fi
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
job_dir="$experiment/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
mkdir -p "$job_dir"
cp -- "$0" "$job_dir/submitted_script.sh"
exec >> "$job_dir/run.log" 2>&1
echo "Started: $(date -Is); host=$(hostname); task=$SLURM_ARRAY_TASK_ID"
echo "CPUs=$SLURM_CPUS_PER_TASK; memory=${SLURM_MEM_PER_NODE:-unknown} MiB"
"$python_bin" -V
"$python_bin" -B -c 'from importlib.metadata import distributions; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in distributions())))' > "$job_dir/packages.txt"
"$python_bin" -B -c 'import numpy; numpy.show_config()' > "$job_dir/numpy_config.txt"
exec /usr/bin/time -v -o "$job_dir/time.txt" \
    "$python_bin" -u -B -m core.annual_jobs run \
    --experiment "$experiment" --task-index "$SLURM_ARRAY_TASK_ID"
