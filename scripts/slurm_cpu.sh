#!/bin/bash
#SBATCH --job-name=ciproseg_train      # job name
#SBATCH --partition=cpuonly            # queue/partition

#SBATCH --nodes=1                      # number of nodes
#SBATCH --ntasks=1                     # total number of tasks
#SBATCH --cpus-per-task=8              # number of CPUs per task
#SBATCH --mem=32G                      # a single total memory request

#SBATCH --time=2-00:00:00   	       # walltime (days-hh:mm:ss)
#SBATCH --account=hk-project-pai00125  # project ID for multiple project account

set -euo pipefail

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

echo "=== CiproSeg Training (CPU) ==="
echo "Timestamp:     ${TIMESTAMP}"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "Node(s):       ${SLURM_JOB_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Project root:  ${PROJECT_ROOT}"
echo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

ls -lah "${PROJECT_ROOT}/configs/experiment.yaml"
ls -lah "${PROJECT_ROOT}/Data/original_size/images"

srun --cpu-bind=cores ciproseg-train --config "${PROJECT_ROOT}/configs/experiment.yaml"
