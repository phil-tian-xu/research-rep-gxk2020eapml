#!/usr/bin/env bash
#SBATCH --job-name=gkx-annual
#SBATCH --account=pg
#SBATCH --partition=scrp
#SBATCH --qos=c32g4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%8
#SBATCH --output=slurm-%A_%a.out

# Submit from the checkout that prepared this experiment.
# sbatch run_annual_scrp.sh /absolute/path/to/prepared/experiment
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "Usage: sbatch run_annual_scrp.sh EXPERIMENT_DIRECTORY" >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
experiment="$1"
if [[ "$experiment" != /* || ! -f "$experiment/experiment.json" ]]; then
    echo "Provide the absolute path of a prepared experiment." >&2
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
exec /usr/bin/time -v -o "$job_dir/time.txt" \
    .venv/bin/python -u -B -m core.annual_jobs run \
    --experiment "$experiment" --task-index "$SLURM_ARRAY_TASK_ID"
