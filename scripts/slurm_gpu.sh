#!/bin/bash
#SBATCH --job-name=ciproseg_train      # job name
#SBATCH --partition=accelerated        # request GPU partition
#SBATCH --gres=gpu:1                   # request 1 GPU

#SBATCH --nodes=1                      # number of nodes
#SBATCH --ntasks=1                     # total number of tasks
#SBATCH --cpus-per-task=4              # number of CPUs per task
#SBATCH --time=2-00:00:00	       # walltime (days-hh:mm:ss)

#SBATCH --output=slurm-%j.out          # output file
#SBATCH --error=slurm-%j.err           # error file
#SBATCH --account=hk-project-pai00125  # project ID for multiple project account

set -euo pipefail

# Resolve PROJECT_ROOT robustly
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "Node(s):       ${SLURM_JOB_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo

if [[ -f "${SLURM_SUBMIT_DIR}/configs/experiment.yaml" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/../configs/experiment.yaml" ]]; then
  PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  echo "ERROR: Could not find configs/experiment.yaml in submit dir or its parent."
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  exit 1
fi

echo "PROJECT_ROOT:  ${PROJECT_ROOT}"
echo

cd "${PROJECT_ROOT}"

OUTDIR=$(python - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/experiment.yaml"))
print((cfg.get("trainer") or {}).get("output_dir", "runs"))
PY
)

TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
LOG_DIR="${PROJECT_ROOT}/${OUTDIR}/logs"
mkdir -p "${LOG_DIR}"

exec > "${LOG_DIR}/Train_${TIMESTAMP}_job${SLURM_JOB_ID}.log" \
     2> "${LOG_DIR}/Train_${TIMESTAMP}_job${SLURM_JOB_ID}.err"

# After exec redirection, re-print a small header into the log
echo "=== CiproSeg Training ==="
echo "Timestamp:     ${TIMESTAMP}"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "Node(s):       ${SLURM_JOB_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo "Project root:  ${PROJECT_ROOT}"
echo

# Thread settings (CPU-side ops)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Sanity checks
ls -lah "${PROJECT_ROOT}/configs/experiment.yaml"
ls -lah "${PROJECT_ROOT}/Data/original_size/images"

echo
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
echo

# Run
srun --cpu-bind=cores ciproseg-train --config "${PROJECT_ROOT}/configs/experiment.yaml"
