#!/usr/bin/env bash
#SBATCH --job-name=gkx-nn-a100
#SBATCH --account=pg
#SBATCH --partition=a100
#SBATCH --qos=c32g4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0%1
#SBATCH --output=slurm-%A_%a.out

# Selected and snapshotted by run_experiment.py nn1/nn2/nn3/nn4/nn5 --scrp.
# The launcher overrides CPUs, RAM, wall time and array indices/concurrency.
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo "Use run_experiment.py NN_MODEL --scrp --sample p05/full --years YEAR/RANGE." >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?}"
experiment="$1"
if [[ "$experiment" != /* || ! -f "$experiment/experiment.json" ]]; then
    echo "Provide the absolute path of a prepared annual experiment." >&2
    exit 2
fi
python_bin="$PWD/.venv/bin/python"
if [[ ! -x "$python_bin" || ! -x /usr/bin/time ]]; then
    echo "Required: project .venv/bin/python and /usr/bin/time." >&2
    exit 2
fi
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
job_dir="$experiment/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
mkdir -p "$job_dir"
cp -- "$0" "$job_dir/submitted_script.sh"
exec >> "$job_dir/run.log" 2>&1
echo "Started: $(date -Is); host=$(hostname); task=$SLURM_ARRAY_TASK_ID"
echo "CPUs=$SLURM_CPUS_PER_TASK; RAM=${SLURM_MEM_PER_NODE:-unknown} MiB; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$job_dir/nvidia_smi.txt"
"$python_bin" -u -B - <<'PY'
import torch

print(f"PyTorch={torch.__version__}; CUDA runtime={torch.version.cuda}", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Check the GPU allocation, driver and 'uv sync --extra nn' installation.")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected one visible GPU, found {torch.cuda.device_count()}.")
print(f"Allocated GPU: {torch.cuda.get_device_name(0)}", flush=True)
inputs = torch.ones((8, 8), device="cuda", requires_grad=True)
(inputs @ inputs).sum().backward()
torch.cuda.synchronize()
print("CUDA matrix forward/backward check passed.", flush=True)
PY
exec /usr/bin/time -v -o "$job_dir/time.txt" \
    "$python_bin" -u -B -m core.annual_jobs run \
    --experiment "$experiment" --task-index "$SLURM_ARRAY_TASK_ID"
