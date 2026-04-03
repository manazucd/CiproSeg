#!/bin/bash
#SBATCH --job-name=ciproseg_train      # job name
#SBATCH --partition=cpuonly            # queue/partition

#SBATCH --nodes=1                      # number of nodes
#SBATCH --ntasks=1                     # total number of tasks
#SBATCH --cpus-per-task=8              # number of CPUs per task
#SBATCH --mem=32G                      # total memory request

#SBATCH --time=2-00:00:00              # walltime (days-hh:mm:ss)

#SBATCH --output=slurm-%j.out          # SLURM stdout
#SBATCH --error=slurm-%j.err           # SLURM stderr
#SBATCH --account=hk-project-pai00125  # project ID

set -euo pipefail

echo "=== CiproSeg Training (CPU) ==="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Partition:     ${SLURM_JOB_PARTITION}"
echo "Node(s):       ${SLURM_JOB_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Submit dir:    ${SLURM_SUBMIT_DIR}"
echo

# Resolve project root
if [[ -f "${SLURM_SUBMIT_DIR}/configs/experiment.yaml" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/../configs/experiment.yaml" ]]; then
  PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
  echo "ERROR: Could not find configs/experiment.yaml"
  exit 1
fi

echo "PROJECT_ROOT:  ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

# Thread settings
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Run
# Training output goes to train.log inside the auto-created run dir:
#   runs/<model>/<timestamp>/train.log
srun --cpu-bind=cores \
  ciproseg-train --config "${PROJECT_ROOT}/configs/experiment.yaml" --cpu
